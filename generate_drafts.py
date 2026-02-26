#!/usr/bin/env python3
import os
import time
import json
import requests
import subprocess

from common import DATA, now_iso, read_jsonl, append_jsonl

IN = DATA / 'articles.jsonl'
HOT = DATA / 'hot_candidates.json'
OUT = DATA / 'drafts.jsonl'
PROCESSED = DATA / 'processed_articles.jsonl'


def llm_prompt(article_text, max_chars, target_format='brief'):
    benches = os.getenv('STYLE_BENCHMARK_ACCOUNTS', 'joo___wol2,info__sum')
    return f"""
너는 한국어 Threads 편집자다.
아래 기사 원문을 바탕으로 결과를 JSON으로만 출력하라.

스타일 참고:
- 벤치마크 계정: {benches}
- 참고는 '구성/리듬/가독성'만 한다.
- 문장/표현/콘텐츠를 복제하지 않는다.

요구사항:
1) title: 한국어 제목 1줄(28자 이내)
2) body: 정보형 + 캐주얼 톤(가벼운 대화체 가능, 비속어 과다 금지)
3) body 길이: {max_chars}자 이내
4) 가독성을 위해 2~4문장으로 분리
5) 해시태그는 넣지 않는다
6) format은 반드시 `{target_format}` 으로 작성한다.
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
    # openclaw can prepend log lines (e.g., tool provider warnings), so trim to first JSON object
    brace = text_out.find('{')
    if brace < 0:
        raise RuntimeError('writer agent: no JSON object in output')
    data = json.loads(text_out[brace:])
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
    done = set((r.get('url') or '') for r in read_jsonl(PROCESSED) if r.get('status') == 'ok')

    hot_only = os.getenv('HOT_ONLY', '1').strip() in ['1', 'true', 'True', 'yes']
    draft_limit = int(os.getenv('DRAFT_LIMIT', str(limit)))

    if hot_only and HOT.exists():
        hot = json.loads(HOT.read_text(encoding='utf-8'))
        hot_urls = [x.get('url') for x in hot.get('items', []) if x.get('url')]
        hot_set = set(hot_urls)
        targets = [r for r in rows if r.get('url') in hot_set and r.get('url') not in done][:draft_limit]
    else:
        targets = [r for r in rows if r.get('url') and r.get('url') not in done][:draft_limit]

    provider = os.getenv('GEN_PROVIDER', 'openai').lower()
    # explicit writer mode: GEN_PROVIDER=writer
    max_chars = int(os.getenv('DRAFT_MAX_CHARS', '250'))
    ok = 0

    format_cycle = ['prep', 'pas', 'listicle', 'whyhow', 'brief', 'insight']

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
            # 해시태그 제거 정책
            body = '\n'.join([ln for ln in body.split('\n') if not ln.strip().startswith('#')]).strip()
            import re
            body = re.sub(r'\s*#[\w가-힣_]+', '', body).strip()
            body = body[:max_chars]
            if not title or not body:
                raise ValueError('empty title/body')

            fmt = (obj.get('format') or '').strip().lower()
            if fmt not in ['prep', 'pas', 'listicle', 'whyhow', 'brief', 'insight']:
                fmt = target_format

            append_jsonl(OUT, {
                'id': f"d_{abs(hash(url))}",
                'ts': now_iso(),
                'url': url,
                'source_title': r.get('source_title', ''),
                'title': title,
                'body': body,
                'format': fmt,
                'provider': used_provider,
                'status': 'draft'
            })
            append_jsonl(PROCESSED, {'ts': now_iso(), 'url': url, 'status': 'ok', 'provider': used_provider, 'format': fmt})
            ok += 1
        except Exception as e:
            append_jsonl(PROCESSED, {'ts': now_iso(), 'url': url, 'status': 'error', 'error': str(e)[:200]})

    print('DRAFT_OK', ok, 'targets', len(targets), 'provider', provider)


if __name__ == '__main__':
    main()
