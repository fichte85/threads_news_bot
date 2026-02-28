#!/usr/bin/env python3
import os
import time
import json
import requests
import subprocess
import re
from difflib import SequenceMatcher
import hashlib
from datetime import datetime, timedelta

from common import DATA, now_iso, read_jsonl, append_jsonl

IN = DATA / 'articles.jsonl'
HOT = DATA / 'hot_candidates.json'
OUT = DATA / 'drafts.jsonl'
PROCESSED = DATA / 'processed_articles.jsonl'
SIMILAR_DAYS = int(os.getenv('DRAFT_SIMILAR_LOOKBACK_DAYS', '3'))


def llm_prompt(article_text, max_chars, target_format='brief'):
    benches = os.getenv('STYLE_BENCHMARK_ACCOUNTS', 'joo___wol2,info__sum')
    return f"""
너는 한국어 Threads 뉴스 편집자다.
아래 기사 원문을 바탕으로 결과를 JSON으로만 출력하라.

스타일 참고:
- 벤치마크 계정: {benches}
- 참고는 '구성/리듬/가독성'만 한다.
- 문장/표현/콘텐츠를 복제하지 않는다.

요구사항(반드시 지킬 것):
1) title: 한국어 제목 1줄(28자 이내)
2) body: 정보형+해석형. 단순 사실 나열이 아니라 '의미와 함의'가 보이게 작성
3) body 길이: {max_chars}자 이내
4) 가독성을 위해 2~4문장(필요 시 줄바꿈)으로 분리
5) 해시태그는 넣지 않는다
6) 라벨(훅:, 근거:, 함의:, 질문:)은 쓰지 말 것.
   - 다만 구조는 유지: 도입(긴장감), 근거(2개 이상), 함의, 마지막 질문형 문장
7) format은 반드시 `{target_format}` 으로 작성한다.
   - allowed: prep, pas, listicle, whyhow, brief, insight

출력 JSON 스키마:
{{"title":"...","body":"...","format":"{target_format}"}}

기사 원문:
{article_text[:7000]}
""".strip()


def ask_openai(prompt):
    key = os.getenv('OPENAI_API_KEY', '')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    r = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={
            'model': model,
            'temperature': 0.7,
            'messages': [
                {'role': 'system', 'content': 'Return valid JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def ask_anthropic(prompt):
    key = os.getenv('ANTHROPIC_API_KEY', '')
    model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-latest')
    r = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        },
        json={
            'model': model,
            'max_tokens': 700,
            'temperature': 0.7,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()['content'][0]['text']


def ask_gemini(prompt):
    key = os.getenv('GEMINI_API_KEY', '')
    model_env = os.getenv('GEMINI_MODEL', 'gemma-3-27b-it')
    # Lite 모델은 RPD 소진 이슈로 기본 fallback에서 제외
    models = [model_env, 'gemma-3-27b-it', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
    # 순서 유지 중복 제거
    models = list(dict.fromkeys(models))
    last_err = None

    for model in models:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
        r = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            json={
                'generationConfig': {'temperature': 0.7},
                'contents': [{'parts': [{'text': prompt}]}],
            },
            timeout=60,
        )
        if r.status_code == 404:
            last_err = RuntimeError(f'Gemini model not found: {model}')
            continue
        r.raise_for_status()
        return r.json()['candidates'][0]['content']['parts'][0]['text']

    raise last_err or RuntimeError('Gemini request failed')


def ask_writer(prompt):
    """Generate via OpenClaw 'writer' agent (에이전트 시드)."""
    # writer는 로컬 에이전트 실행 플래그로 호출해 API 키/토큰 의존성에서 벗어나도록 구성
    cmd = [
        'openclaw', 'agent',
        '--agent', 'writer',
        '--local',
        '--json',
        '--message', prompt,
    ]
    proc = subprocess.run(
        cmd,
        cwd='/home/ubuntu/.openclaw',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or '').strip()[:500]
        raise RuntimeError(f'writer agent failed: code={proc.returncode} err={err}')

    text_out = proc.stdout.strip()
    m = re.search(r'\{[\s\S]*\}', text_out)
    if not m:
        raise RuntimeError('writer agent: no JSON object in output')
    data = json.loads(m.group(0))
    payloads = data.get('payloads', [])
    if not payloads:
        raise RuntimeError('writer agent: no payloads in response')
    text = payloads[0].get('text', '')
    return text


def parse_json_text(s):
    import json, re
    m = re.search(r'\{[\s\S]*\}', s)
    if not m:
        raise ValueError('JSON not found')
    return json.loads(m.group(0))

BAD_PHRASES = [
    '접속 실패', '접근 실패', '접근 차단', '접근 제한', '접근거부', '접근 거부',
    '오류', '서버 오류', '연결 실패', '로봇 체크', '요청하신 페이지를 찾을 수 없습니다',
    '모더레이션', 'mod_security', '차단', '차단됨', '서버 오류 발생', '자동화 방지',
    '보안 차단', '접근이 차단', '로봇 인증', 'captcha', '캡차', 'bot check', '접근이 거부'
]


def is_blocked_raw(row):
    txt = ((row.get('source_title') or '') + (row.get('text') or '') + (row.get('url') or '') +
           (row.get('resolved_url') or '') + (row.get('final_url') or '') + (row.get('body') or '')).lower()
    if '블룸버그' in txt or 'bloomberg' in txt:
        return False
    return any((p in txt) for p in BAD_PHRASES)


def normalize_candidates(rows):
    return [r for r in rows if not is_blocked_raw(r)]





def strip_section_labels(text: str) -> str:
    return re.sub(r'(?m)^\s*(훅|근거|함의|질문)\s*:\s*', '', (text or ''))


def _norm_kr_text(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-zA-Z0-9가-힣\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _tokenize(s):
    return set(_norm_kr_text(s).split())


def _similar_enough(text1, text2, threshold=0.82):
    a = _norm_kr_text(text1)
    b = _norm_kr_text(text2)
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def is_similar_draft(candidate, existing_rows):
    c_title = candidate.get('title') or candidate.get('source_title') or ''
    c_body = candidate.get('text') or ''
    if not c_title and not c_body:
        return False

    ct = _norm_kr_text(c_title)
    cb_tokens = _tokenize(c_body) | _tokenize(c_title)
    for r in existing_rows:
        e_title = (r.get('title') or '')
        e_body = (r.get('body') or '')
        if _similar_enough(ct, (e_title or '').lower()):
            return True
        et = _norm_kr_text(e_title)
        etokens = _tokenize(e_title) | _tokenize(e_body)
        if cb_tokens and etokens:
            inter = len(cb_tokens & etokens)
            union = len(cb_tokens | etokens)
            if union >= 4 and inter / union >= 0.76:
                return True
    return False


def filter_similar_candidates(rows, existing_rows):
    seen = set()
    out = []
    for r in rows:
        rid = r.get('url') or r.get('title') or ''
        if rid in seen:
            continue
        seen.add(rid)

        # 제목/본문 유사도로 이미 올라간 기사 건너뜀
        if is_similar_draft(r, existing_rows):
            continue
        out.append(r)
    return out

def parse_iso_to_dt(s):
    try:
        if not s:
            return None
        v = str(s).replace('Z', '+00:00')
        return datetime.fromisoformat(v)
    except Exception:
        return None


def recent_urls_set(rows, ttl_days):
    if ttl_days <= 0:
        return set()
    now = datetime.now()
    cut = now - timedelta(days=ttl_days)
    out = set()
    for r in rows:
        if (r.get('status') or '') != 'ok':
            continue
        dt = parse_iso_to_dt(r.get('ts'))
        if dt and dt >= cut:
            out.add(r.get('url'))
    return out


def generate_with_provider(provider, prompt):
    if provider == 'anthropic':
        return ask_anthropic(prompt), provider
    if provider == 'gemini':
        return ask_gemini(prompt), provider
    if provider == 'writer':
        return ask_writer(prompt), 'writer'
    return ask_openai(prompt), 'openai'


def main(limit=30):
    rows = read_jsonl(IN)
    drafted_rows = read_jsonl(OUT)
    rewrite_queued = os.getenv('REWRITE_QUEUED_DRAFTS', '0').strip().lower() in ['1', 'true', 'yes', 'on']
    drafted_urls = {
        (r.get('url') or '') for r in drafted_rows
        if (r.get('status') or '').lower() in ('draft', 'archived', 'published', 'posted')
        or ((r.get('status') or '').lower() in ('queued',) and not rewrite_queued)
    }

    processed_rows = read_jsonl(PROCESSED)
    processed_ttl_days = int(os.getenv('PROCESSED_TTL_DAYS', '0'))
    # 기본은 이전 동작 유지(이미 발행/완료로 처리된 기사 재생성 방지)
    # TTL>0이면 최근 기간 내 'ok'만 제외
    done_recent = recent_urls_set(processed_rows, processed_ttl_days)
    allow_retry = os.getenv('ALLOW_REGENERATE_PROCESSED_DRAFTS', '1').strip().lower() in ['1', 'true', 'yes', 'on']

    hot_only = os.getenv('HOT_ONLY', '1').strip() in ['1', 'true', 'True', 'yes']
    draft_limit = int(os.getenv('DRAFT_LIMIT', str(limit)))

    if hot_only and HOT.exists():
        hot = json.loads(HOT.read_text(encoding='utf-8'))
        hot_urls = [x.get('url') for x in hot.get('items', []) if x.get('url')]
        hot_set = set(hot_urls)

        base = [r for r in rows if r.get('url') in hot_set]
        base = normalize_candidates(base)
        fresh = [r for r in base if r.get('url') not in done_recent]
        if allow_retry:
            targets = fresh[:draft_limit]
            if len(targets) < draft_limit:
                # TTL 내 중복 제외한 나머지까지 보강
                remain = []
                used = set((r.get('url') or '') for r in targets)
                for r in base:
                    url = r.get('url') or ''
                    if url in used:
                        continue
                    if url in done_recent:
                        remain.append(r)
                    if len(targets) + len(remain) >= draft_limit:
                        break
                targets.extend(remain[:max(0, draft_limit - len(targets))])
        else:
            targets = fresh[:draft_limit]
    else:
        fresh_all = [r for r in rows if r.get('url') and r.get('url') not in done_recent]
        fresh_all = normalize_candidates(fresh_all)
        if allow_retry:
            targets = fresh_all[:draft_limit]
        else:
            targets = fresh_all[:draft_limit]

    # 기존 큐/기존 초안/발행 히스토리 URL 중복 제거
    targets = [r for r in targets if (r.get('url') or '') not in drafted_urls]

    # 최근 생성 성공본(lookback)과의 유사도 중복도 차단
    recent_ok_urls = recent_urls_set(processed_rows, SIMILAR_DAYS)
    recent_article_rows = [
        {
            'title': (r.get('source_title') or ''),
            'body': (r.get('text') or ''),
            'url': (r.get('url') or ''),
        }
        for r in rows
        if (r.get('url') or '') in recent_ok_urls
    ]

    # queued 재생성 모드에서는 최근 유사도 가드를 완화해 기존 큐 항목을 즉시 교체
    force_rewrite_queued = rewrite_queued and os.getenv('SKIP_SIMILARITY_FOR_REWRITE', '1').strip().lower() in ['1', 'true', 'yes', 'on']

    target_urls = [x.strip() for x in os.getenv('TARGET_DRAFT_URLS', '').split(',') if x.strip()]
    if target_urls:
        target_set = set(target_urls)
        targets = [r for r in targets if (r.get('url') or '') in target_set]

    # archived/draft/queued/published 및 최근 생성본과 유사한 후보 제거
    if force_rewrite_queued:
        # rewrite 모드: 최근 유사도 필터를 건너뛰고, 기존 queued 대상만 처리
        pass
    else:
        similarity_base = drafted_rows + recent_article_rows
        targets = filter_similar_candidates(targets, similarity_base)

    provider = os.getenv('GEN_PROVIDER', 'openai').lower()
    # explicit writer mode: GEN_PROVIDER=writer
    max_chars = int(os.getenv('DRAFT_MAX_CHARS', '250'))
    ok = 0
    recent_generated = []

    format_cycle = ['insight', 'whyhow', 'brief', 'prep', 'pas', 'listicle']

    for idx, r in enumerate(targets):
        url = r['url']
        try:
            target_format = format_cycle[idx % len(format_cycle)]
            prompt = llm_prompt(r.get('text', ''), max_chars, target_format=target_format)
            used_provider = provider
            try:
                raw, used_provider = generate_with_provider(provider, prompt)
            except Exception as e:
                # OpenAI 429 등 레이트리밋 시 Gemini로 1회 폴백 (writer 모드는 폴백하지 않음)
                if provider == 'openai' and '429' in str(e) and os.getenv('GEMINI_API_KEY', ''):
                    time.sleep(1.2)
                    raw, used_provider = generate_with_provider('gemini', prompt)
                else:
                    raise

            obj = parse_json_text(raw)
            title = (obj.get('title') or '').strip()[:28]
            body = (obj.get('body') or '').strip()
            if is_blocked_raw({'source_title': title, 'text': body}):
                raise ValueError('blocked content pattern')
            # 해시태그/섹션 라벨 제거 정책
            body = '\n'.join([ln for ln in body.split('\n') if not ln.strip().startswith('#')]).strip()
            import re
            body = re.sub(r'\s*#[\w가-힣_]+', '', body).strip()
            body = strip_section_labels(body).strip()

            # 라벨을 제거했더라도 마지막은 질문형으로 마무리(혹시 안 맞는 경우 보정)
            question = '어떤 결론이 더 납득되시나요?'
            if body and not body.rstrip().endswith('?'):
                reserve = 1 + len(question)  # "\n" + question
                if len(body) + reserve <= max_chars:
                    body = f"{body}\n{question}"
                else:
                    body = body[: max_chars - reserve]
                    body = body.rstrip()
                    if not body:
                        body = question
                    else:
                        body = f"{body}\n{question}"

            body = body[:max_chars]
            if not title or not body:
                raise ValueError('empty title/body')
            # 최종 보정: 잘림으로 질문이 사라진 경우 강제 마무리
            if not body.rstrip().endswith('?'):
                q = '어떤 게 더 맞을까요?'
                if len(q) + 1 >= max_chars:
                    body = q[:max_chars]
                else:
                    body = (body[: max_chars - (len(q) + 1)]).rstrip() + '\n' + q

            # 생성본이 기존/최근 생성본과 유사하면 건너뜀 (큐 즉시 갱신 모드에서는 기존 queued 덮어쓰기에 방해되지 않도록 예외 허용)
            candidate_preview = {'title': title, 'body': body}
            if not force_rewrite_queued:
                if is_similar_draft(candidate_preview, drafted_rows + recent_generated):
                    raise ValueError('duplicate-like draft content')

            fmt = (obj.get('format') or '').strip().lower()
            if fmt not in ['prep', 'pas', 'listicle', 'whyhow', 'brief', 'insight']:
                fmt = target_format

            draft_item = {
                'id': f"d_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
                'ts': now_iso(),
                'url': url,
                'source_title': r.get('source_title', ''),
                'title': title,
                'body': body,
                'format': fmt,
                'provider': used_provider,
                'status': 'draft'
            }
            if rewrite_queued:
                # 같은 URL의 기존 queued 초안 덮어쓰기
                drafted_rows = [x for x in drafted_rows if not ((x.get('url') or '') == url and (x.get('status') or '').lower() == 'queued')]
                drafted_rows.append(draft_item)
            else:
                append_jsonl(OUT, draft_item)
            recent_generated.append({'title': title, 'body': body})
            append_jsonl(PROCESSED, {'ts': now_iso(), 'url': url, 'status': 'ok', 'provider': used_provider, 'format': fmt})
            ok += 1
        except Exception as e:
            append_jsonl(PROCESSED, {'ts': now_iso(), 'url': url, 'status': 'error', 'error': str(e)[:200]})

    # rewrite mode: queued 초안만 갱신하는 경우 변경된 drafted_rows를 전체 파일로 반영
    if rewrite_queued:
        with OUT.open('w', encoding='utf-8') as f:
            for r in drafted_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('DRAFT_OK', ok, 'targets', len(targets), 'provider', provider, 'rewrite_queued', rewrite_queued)


if __name__ == '__main__':
    main()
