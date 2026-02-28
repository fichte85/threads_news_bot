#!/usr/bin/env python3
import os, time, datetime, json, requests, subprocess, re, threading
from pathlib import Path
from common import DATA, read_jsonl, read_json, write_json, update_json_locked

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = str(os.getenv('TELEGRAM_CHAT_ID', ''))

DRAFTS = DATA / 'drafts.jsonl'
QUEUE = DATA / 'publish_queue.json'
HOT = DATA / 'hot_candidates.json'

last_update_id = 0
ui_state = {
    'mode': None,            # None | 'pick_queue_item' | 'pick_template'
    'queue_idx': None,
}

# 비동기 작업 상태
JOB_DEFINITIONS = {
    'collect_extract': {
        'cmd': 'python3 rss_collect.py && python3 extract_articles.py',
        'label': '수집/추출',
    },
    'generate_news': {
        'cmd': 'python3 hot_only.py && python3 generate_drafts.py',
        'label': '초안 생성',
    },
}

background_jobs = {}

# 혼합 모드 단기 자동해제(메모리 카운터)
MIX_AUTO_OFF_REMAIN = 0

SENSITIVE_PATTERNS = [
    ('auth_header', re.compile(r'authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._\-]{8,}', re.IGNORECASE)),
    ('bearer_token', re.compile(r'\bbearer\s+[A-Za-z0-9._\-]{8,}', re.IGNORECASE)),
    ('api_key_wording', re.compile(r'\b(?:token|api(?:[_-]?key)?|access\s*key|secret\s*key)\b\s*[:=]?\s*[A-Za-z0-9._\-/]{8,}', re.IGNORECASE)),
    ('query_key_token', re.compile(r'[?&](?:token|access_token|id_token|auth|authorization|api(?:[_-]?key)?|key|secret|sig|signature|session(?:id)?|code)=[^\s&#]{8,}', re.IGNORECASE)),
    ('prefix_sk', re.compile(r'\bsk-[A-Za-z0-9]{10,}\b')),
    ('prefix_xoxb', re.compile(r'\bxoxb-[A-Za-z0-9-]{10,}\b')),
    ('prefix_ghp', re.compile(r'\bghp_[A-Za-z0-9]{10,}\b')),
    ('prefix_akia', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('prefix_aiza', re.compile(r'\bAIza[0-9A-Za-z\-_]{20,}\b')),
    ('prefix_ya29', re.compile(r'\bya29\.[0-9A-Za-z\-_]+\b')),
    ('email', re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
    ('phone_kr', re.compile(r'\b(?:\+?82[-\s]?)?0?1[016789][-\s]?\d{3,4}[-\s]?\d{4}\b')),
    ('long_handle', re.compile(r'(?<!\w)@[A-Za-z0-9_.-]{12,}')),
    ('url_query_suspect', re.compile(r'https?://[^\s?#]+\?[^\s]{20,}(?:token|access_token|id_token|auth|api(?:[_-]?key)?|key|secret|sig|code)', re.IGNORECASE)),
]

SAFE_CHAT_FALLBACK = '민감정보가 감지되어 내용을 일반화했어요. 이 이슈에서 어떤 관점이 가장 중요하다고 보세요?'
SAFE_REWRITE_FALLBACK = '민감정보 보호 정책에 따라 내용을 일반화했습니다. 핵심 이슈와 공개 가능한 사실 중심으로 다시 정리해 주세요.'


def detect_sensitive_pattern(text):
    txt = str(text or '')
    for name, pattern in SENSITIVE_PATTERNS:
        if pattern.search(txt):
            return name
    return None


def sanitize_generated_text(text, *, context, regenerate_fn=None, max_retries=2, fallback_text=''):
    """생성 직후 민감정보 패턴을 차단한다. initial + max_retries(최대 2회) 후 fallback."""
    current = str(text or '').strip()
    attempt = 0
    while True:
        pattern_name = detect_sensitive_pattern(current)
        if not pattern_name:
            return current

        print(f"reason=sensitive-block:pattern={pattern_name} context={context} attempt={attempt}")
        if regenerate_fn is not None and attempt < max_retries:
            attempt += 1
            try:
                current = str(regenerate_fn(attempt, pattern_name) or '').strip()
                continue
            except Exception as e:
                print(f"reason=sensitive-block:pattern=regenerate-error context={context} error={str(e)[:120]}")
        print(f"reason=sensitive-block:pattern={pattern_name} context={context} action=fallback")
        return fallback_text or ''


def send(text):
    if not TOKEN or not CHAT_ID:
        return
    keyboard = {
        'keyboard': [
            [{'text': '뉴스 수집'}, {'text': '뉴스생성'}, {'text': '뉴스 리뷰'}],
            [{'text': '스케쥴관리'}, {'text': '잡담'}, {'text': '상태'}],
        ],
        'resize_keyboard': True,
        'one_time_keyboard': False,
    }
    requests.post(
        f'https://api.telegram.org/bot{TOKEN}/sendMessage',
        json={'chat_id': CHAT_ID, 'text': text, 'reply_markup': json.dumps(keyboard, ensure_ascii=False)}, timeout=10
    )


def _publish_env_for_news2(cdp_url=None, own_handle=None):
    """news2 publish용 Node env 재현 + 모듈 경로를 고정해 충돌을 줄인다."""
    env = os.environ.copy()
    if cdp_url:
        env['THREADS_CDP_URL'] = cdp_url
    if own_handle:
        env['THREADS_OWN_HANDLE'] = own_handle

    nm_candidates = [
        '/home/ubuntu/threads-bot/node_modules',
        '/home/ubuntu/threads-bot-news2/node_modules',
    ]
    node_path = env.get('NODE_PATH', '')
    for p in nm_candidates:
        if Path(p).exists():
            node_path = f"{p}:{node_path}" if node_path else p
            break
    env['NODE_PATH'] = node_path
    env['THREADS_PUBLISH_LOG'] = os.getenv('THREADS_PUBLISH_LOG', '/home/ubuntu/threads-bot-news2/threads_publish_full.log')
    return env


def run(cmd):
    r = subprocess.run(['bash', '-lc', cmd], cwd='/home/ubuntu/threads-bot-news2', capture_output=True, text=True, timeout=None)
    return r.returncode, (r.stdout + '\n' + r.stderr)[-1000:]


def run_publish_news2(publish_js, payload, own_handle='', cdp_url='http://127.0.0.1:9223/'):
    env = _publish_env_for_news2(cdp_url=cdp_url, own_handle=own_handle)
    r = subprocess.run(
        ['node', publish_js, payload],
        cwd='/home/ubuntu/threads-bot-news2',
        env=env,
        capture_output=True,
        text=True,
        timeout=None,
    )
    return r.returncode, (r.stdout + '\n' + r.stderr)[-1000:]


def run_async_job(job_name, cmd, success_cb=None):
    existing = background_jobs.get(job_name)
    if existing and existing.get('thread') and existing['thread'].is_alive():
        return False, f"이미 {existing.get('label')} 작업이 진행 중입니다. /status로 확인하세요."

    job = {
        'started': time.time(),
        'label': JOB_DEFINITIONS.get(job_name, {}).get('label', job_name),
        'status': 'running',
        'exit_code': None,
        'output': '',
        'thread': None,
    }

    def _worker():
        code, out = run(cmd)
        job['exit_code'] = code
        job['output'] = out
        job['status'] = 'done'
        job['finished'] = time.time()
        if success_cb:
            try:
                success_cb(code, out)
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    job['thread'] = t
    background_jobs[job_name] = job
    t.start()
    return True, f"{job['label']} 작업을 백그라운드로 시작했어요. 완료되면 자동으로 알림할게요."




def is_blocked_draft_item(d):
    bad_phrases = [
        '접속 실패', '접근 실패', '접근 차단', '접근 제한', '오류', '서버 오류', '연결 실패', '로봇 체크',
        '요청하신 페이지를 찾을 수 없습니다', '모더레이션', 'mod_security', '차단', '서버 오류 발생'
    ]
    txt = ((d.get('title') or '') + (d.get('body') or '') + (d.get('text') or '') + (d.get('summary') or '')).lower()

    # Bloomberg 정책(최종):
    # - Bloomberg + 로봇체크/차단 문구는 운영상 예외로 review/show/showq에서 확인 가능하도록 보존
    # - 그 외 일반 차단/오류 문구는 기존대로 필터
    if ('블룸버그' in txt or 'bloomberg' in txt) and any(k in txt for k in ['로봇 체크', 'robot', 'captcha', '차단']):
        return False

    return any(b in txt for b in bad_phrases)

def pending_drafts():
    rows = read_jsonl(DRAFTS)
    return [r for r in rows if (r.get('status') or 'draft') == 'draft' and not is_blocked_draft_item(r)]


def hot_map_by_url():
    d = read_json(HOT, {'items': []})
    m = {}
    for it in d.get('items', []):
        u = str(it.get('url') or '')
        if u:
            m[u] = it
    return m


def archive_pending_drafts():
    rows = read_jsonl(DRAFTS)
    changed = 0
    for r in rows:
        if (r.get('status') or 'draft') == 'draft':
            r['status'] = 'archived'
            changed += 1
    if changed:
        with open(DRAFTS, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return changed


def build_news_slots(n, start_from=None):
    # 뉴스는 2시간 간격 고정(정규 슬롯)
    if start_from is None:
        start = os.getenv('PUBLISH_START', '10:30')
        sh, sm = map(int, start.split(':'))
        now = datetime.datetime.now()
        day = now.date()
        base = datetime.datetime.combine(day, datetime.time(sh, sm))
        # 이미 지난 시간대면 다음 날 시작시간부터 잡는다.
        if base <= now:
            day = day + datetime.timedelta(days=1)
            base = datetime.datetime.combine(day, datetime.time(sh, sm))
    else:
        if isinstance(start_from, (int, float)):
            base = datetime.datetime.fromtimestamp(start_from)
        else:
            base = start_from
        # 분/초를 버리고, 다음 즉시 슬롯으로 정렬
        base = base.replace(second=0, microsecond=0)
        if datetime.datetime.now() > base:
            base = base + datetime.timedelta(minutes=1)

    out = [base + datetime.timedelta(hours=2 * i) for i in range(n)]
    return out


def merge_with_chat_schedule(news_items, chat_items, start_from=None):
    # news_items/chat_items: [{'id','text','template?'}]
    news_slots = build_news_slots(len(news_items), start_from=start_from)
    scheduled_news = []
    for it, t in zip(news_items, news_slots):
        scheduled_news.append({
            'id': it['id'],
            'when': t.strftime('%Y-%m-%d %H:%M'),
            'text': it['text'],
            'template': (it.get('template') or 'news')
        })

    # chat은 뉴스 사이 비정규 시간에 배치
    scheduled_chat = []
    offsets = [67, 49, 73, 41, 58]
    if len(scheduled_news) >= 2:
        for i, ch in enumerate(chat_items):
            anchor = min(2 + i, len(scheduled_news) - 2)
            t1 = datetime.datetime.strptime(scheduled_news[anchor]['when'], '%Y-%m-%d %H:%M')
            t2 = datetime.datetime.strptime(scheduled_news[anchor + 1]['when'], '%Y-%m-%d %H:%M')
            cand = t1 + datetime.timedelta(minutes=offsets[i % len(offsets)])
            if cand >= t2:
                cand = t1 + datetime.timedelta(minutes=50)
            scheduled_chat.append({
                'id': ch['id'],
                'when': cand.strftime('%Y-%m-%d %H:%M'),
                'text': ch['text'],
                'template': 'chat'
            })

    merged = scheduled_news + scheduled_chat
    merged.sort(key=lambda x: x['when'])
    return merged


def _extract_short_topic(text, limit=3):
    t = (text or '').lower()
    t = re.sub(r'[^0-9a-z가-힣 ]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    stop = {
        'the', 'a', 'an', 'of', 'to', 'in', 'for', 'and', 'on', 'at', 'is', 'are', 'was', 'were',
        'this', 'that', 'it', 'as', 'by', 'with', 'from', 'today', 'news', 'latest', 'yesterday', 'market',
        'economy', 'tech', 'report', 'update', 'breaking', 'update',
    }
    words = []
    for w in t.split():
        if len(w) <= 1 or w in stop:
            continue
        words.append(w)
    out = []
    for w in words:
        if w not in out:
            out.append(w)
    return out[:limit]



def _topic_category(words):
    if not words:
        return '핵심이슈'
    return words[0]


def _keyword_topics(text, limit=4):
    words = _extract_short_topic(text or '', limit=limit * 2)
    ban = {
        '지금', '오늘', '이번', '관련', '대한', '통해', '정도', '이후', '최근', '가장', '정리',
        '한국', '미국', '중국', '시장', '정책', '발표', '가능성', '이슈', '핵심',
    }
    topics = []
    for w in words:
        if len(w) <= 1 or w in ban:
            continue
        if w not in topics:
            topics.append(w)
    return topics[:limit]


def _topic_phrase(seed_text='', prev_text='', next_text=''):
    merged = ' '.join([seed_text or '', prev_text or '', next_text or '']).strip()
    topics = _keyword_topics(merged, limit=3)
    if not topics:
        return '이 흐름'
    if len(topics) == 1:
        return topics[0]
    return f"{topics[0]}·{topics[1]}"


def _emotion_level(seed_text=''):
    t = (seed_text or '').lower()
    hard = ['하락', '급락', '충격', '리스크', '전쟁', '규제', '위기', '불안', '분쟁', '실적부진', '악화']
    bright = ['상승', '호재', '기대', '회복', '돌파', '확대', '개선', '신기록', '반등']
    hard_score = sum(1 for k in hard if k in t)
    bright_score = sum(1 for k in bright if k in t)
    if hard_score >= 2:
        return 'low'
    if bright_score >= 2:
        return 'high'
    return 'mid'


def _shape_text_len(text, level='mid'):
    # low: 차분/짧게, high: 에너지 조금 높게, mid: 기본
    if not text:
        return text
    target = {'low': 120, 'mid': 145, 'high': 165}.get(level, 145)
    t = re.sub(r'\s+', ' ', text).strip()
    if len(t) <= target:
        return t
    cut = t[:target].rstrip(' ,.;')
    # 문장 끝 정리
    if not re.search(r'[.!?…]$', cut):
        cut += '…'
    return cut


def _is_too_similar(cand, existing_texts):
    c = re.sub(r'\s+', ' ', (cand or '').strip())
    if not c:
        return True
    c_words = set(c.split())
    for ex in existing_texts or []:
        e = re.sub(r'\s+', ' ', str(ex).strip())
        if not e:
            continue
        if c == e:
            return True
        e_words = set(e.split())
        if not e_words:
            continue
        overlap = len(c_words & e_words) / max(1, min(len(c_words), len(e_words)))
        if overlap >= 0.78:
            return True
    return False


def make_chitchat(seed_text='', recent_texts=None, prev_news_text='', next_news_text='', force_question=None):
    """뉴스 주제/감정에 반응하는 공감형 잡담 생성기.
    - 뉴스 문맥 키워드 반영(맥락 불일치 완화)
    - 질문형 비율 강제(저관여 문장 방지)
    - 감정 레벨/문장 길이 자동 보정
    - 유사도 기반 중복 회피"""
    seed_bits = seed_text or ''
    recent_texts = recent_texts or []

    warmth = (os.getenv('CHAT_WARMTH', 'high') or 'high').strip().lower()  # low|mid|high
    ask_ratio = float(os.getenv('CHAT_QUESTION_RATIO', '0.88') or 0.88)
    ask_ratio = min(1.0, max(0.0, ask_ratio))

    topic = _topic_phrase(seed_text, prev_news_text, next_news_text)
    emo = _emotion_level(' '.join([seed_text or '', prev_news_text or '', next_news_text or '']))

    empathy_openers = {
        'low': [
            f'{topic} 이슈라 마음이 좀 무거울 수 있어요.',
            f'{topic} 얘기는 체감 피로도가 꽤 높죠.',
            f'{topic} 흐름 보니 신중하게 보는 게 맞아 보여요.',
        ],
        'mid': [
            f'{topic} 쪽이 지금 흐름을 꽤 좌우하네요.',
            f'{topic} 포인트는 그냥 지나치기 어렵죠.',
            f'{topic} 이슈, 생각보다 파급이 길어질 수 있겠어요.',
        ],
        'high': [
            f'{topic} 흐름, 생각보다 재밌게 전개되네요.',
            f'{topic} 건은 분위기 전환 신호로 보이기도 해요.',
            f'{topic} 이슈가 판을 살짝 바꾸는 느낌이에요.',
        ],
    }

    bodies = {
        'low': [
            '지금은 단정하기보다, 리스크가 어디에 쌓이는지 먼저 보는 편이 안전해요.',
            '속도보다 방향 확인이 더 중요해 보여요.',
            '한 번에 결론내리기보다 신호를 한두 개 더 확인하는 게 좋아요.',
        ],
        'mid': [
            '핵심은 숫자보다 의도와 지속 가능성 같아요.',
            '표면 뉴스보다 다음 선택지가 어떻게 바뀌는지가 더 중요해요.',
            '당장 결과보다 구조 변화 쪽이 오래 남는 포인트예요.',
        ],
        'high': [
            '이번엔 단기 반응보다 중기 시나리오를 같이 보는 게 더 재밌어요.',
            '한 번의 이벤트보다 누적 신호가 더 크게 작동할 수 있어요.',
            '오히려 이런 구간에서 관점 차이가 확실히 드러나요.',
        ],
    }

    questions = [
        f'당신은 {topic} 이슈에서 지금 가장 먼저 봐야 할 포인트를 뭐로 봐요?',
        f'{topic} 관련해서, 단기 반응이랑 구조 변화 중 어디에 더 무게를 둘래요?',
        f'이 흐름에서 한 가지 지표만 고른다면 뭘 체크하겠어요?',
        f'지금 판단을 바꿀 수 있는 신호가 하나 더 나온다면 어떤 종류일까요?',
    ]

    closers = [
        '서두르지 말고, 핵심 신호 한두 개만 붙잡고 가도 충분해요.',
        '작게 정리해도 좋아요. 관점을 유지하는 게 이기는 방법이니까요.',
        '결론을 늦추는 것도 전략이에요. 지금은 그게 더 현실적일 수 있어요.',
    ]

    key = warmth if warmth in empathy_openers else 'mid'
    base = abs(hash(seed_bits or str(datetime.datetime.now())))

    def pick(pool, salt):
        return pool[(base + salt) % len(pool)]

    candidates = []
    for i in range(14):
        opener = pick(empathy_openers.get(emo, empathy_openers[key]), i)
        body = pick(bodies.get(emo, bodies[key]), i * 3 + 1)
        ask = force_question if force_question is not None else (((base + i) % 100) < int(ask_ratio * 100))
        tail = pick(questions, i * 5 + 2) if ask else pick(closers, i * 7 + 4)
        text = _shape_text_len(f"{opener} {body} {tail}", level=emo)
        candidates.append(text)

    for cand in candidates:
        if not _is_too_similar(cand, recent_texts):
            return cand

    return candidates[0]



def parse_queue_dt(v):
    return datetime.datetime.strptime(v, '%Y-%m-%d %H:%M')


def insert_chats_into_queue(count=1):
    """현재 큐 기준으로 뉴스 사이에 잡담을 삽입"""
    try:
        count = int(count)
    except Exception:
        return '잡담 개수는 숫자여야 합니다. 예: /chat 2'
    if count <= 0:
        return '잡담 개수는 1 이상으로 주세요.'

    def _mutator(q):
        items = q.get('items', [])
        if not items:
            return q

        # existing chat texts for dedupe
        existing_chat_texts = {
            it.get('text', '')
            for it in items
            if (it.get('template') or '') == 'chat'
        }

        news = []
        for it in items:
            if (it.get('template') or 'news') == 'chat':
                continue
            try:
                dt = parse_queue_dt(it.get('when', ''))
            except Exception:
                continue
            news.append((dt, it))

        if len(news) < 2:
            return q

        news.sort(key=lambda x: x[0])
        cands = []
        for i in range(len(news) - 1):
            dt1, it1 = news[i]
            dt2, it2 = news[i + 1]
            gap = dt2 - dt1
            if gap <= datetime.timedelta(minutes=1):
                continue
            cands.append((gap, dt1 + (gap / 2), it1.get('text', ''), it2.get('text', '')))

        if not cands:
            return q

        # 큰 간격 우선 정렬 후, 호출 이력 커서를 이용해 같은 위치 반복을 피한다.
        cands.sort(key=lambda x: x[0], reverse=True)
        cursor = int(q.get('__chat_insert_cursor', 0) or 0)
        if cursor < 0:
            cursor = 0

        inserted = 0
        for offset in range(count):
            idx = (cursor + offset) % len(cands)
            _gap, dt, ptxt, ntx = cands[idx]
            when = dt.replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')
            # 중복 시간 회피
            probe = dt
            for _ in range(120):
                if any((i.get('when') == when) for i in items):
                    probe = probe + datetime.timedelta(minutes=1)
                    when = probe.replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')
                else:
                    break

            # 같은 텍스트 반복 방지: 슬롯/시점/순번을 시드에 반영
            base_seed = f"{ptxt[:20]}|{ntx[:20]}|{when}|{idx}|{len(items)}"
            ctext = make_chitchat(
                base_seed,
                list(existing_chat_texts),
                prev_news_text=ptxt,
                next_news_text=ntx,
                force_question=True,
            )
            retry = 0
            while ctext in existing_chat_texts and retry < 8:
                retry += 1
                ctext = make_chitchat(
                    base_seed + f"|{retry}",
                    list(existing_chat_texts),
                    prev_news_text=ptxt,
                    next_news_text=ntx,
                    force_question=True,
                )

            ctext = sanitize_generated_text(
                ctext,
                context=f'chat-insert:slot={idx}:when={when}',
                regenerate_fn=lambda retry, _pattern: make_chitchat(
                    base_seed + f"|sensitive|{retry}|{time.time_ns()}",
                    list(existing_chat_texts),
                    prev_news_text=ptxt,
                    next_news_text=ntx,
                    force_question=True,
                ),
                max_retries=2,
                fallback_text=SAFE_CHAT_FALLBACK,
            )

            items.append({
                'id': f'chat_{int(time.time())}_{abs(hash(when)) % 10000000}',
                'when': when,
                'text': ctext,
                'template': 'chat',
            })
            existing_chat_texts.add(ctext)
            inserted += 1

        if inserted:
            items.sort(key=lambda x: x.get('when', '0000-00-00 00:00'))
            q['__chat_insert_cursor'] = (cursor + inserted) % len(cands)
        else:
            q['__chat_insert_cursor'] = cursor

        q['items'] = items
        q['__inserted_chats'] = inserted
        return q

    result = update_json_locked(QUEUE, {'items': []}, _mutator)
    inserted = int((result or {}).get('__inserted_chats', 0))
    return f'현재 큐에 잡담을 {inserted}개 삽입했습니다.' if inserted else '삽입 가능한 구간이 부족해요. 큐가 비었거나 뉴스가 1개뿐일 수 있어요.'


def _normalize_queue_items(items):
    return sorted(items, key=lambda x: x.get('when', '0000-00-00 00:00'))


def move_queue_item(src_idx, dst_idx):
    """1-based 큐 index 이동.
    - src_idx: 이동할 원본 위치
    - dst_idx: 이동될 위치(시간상 정렬 기준).
    src를 제거하고, dst 위치 슬롯의 when을 그 시간으로 설정해 재삽입
    """
    from common import read_json
    data = read_json(QUEUE, {'items': []})
    items = data.get('items', [])
    if not isinstance(items, list):
        return '큐 형식이 손상되어 수정할 수 없습니다.'

    n = len(items)
    if not (1 <= src_idx <= n) or not (1 <= dst_idx <= n):
        return f'큐 범위를 벗어났습니다. 현재 항목: {n}개'
    if src_idx == dst_idx:
        return f'{src_idx}번과 {dst_idx}번이 동일해요. 변경할 필요가 없습니다.'

    src_i = src_idx - 1
    dst_i = dst_idx - 1
    item = items[src_i]

    if src_i > dst_i:
        # remove from left then insert position adjusted
        target_when = items[dst_i].get('when')
    else:
        target_when = items[dst_i].get('when')

    removed = items.pop(src_i)

    # 재삽입 위치(현재 상태에서의 정렬 대상) 결정
    now_len = len(items)
    items.append(removed)
    removed['when'] = target_when

    data['items'] = _normalize_queue_items(items)
    write_json(QUEUE, data)
    return (
        f'{src_idx}번 항목을 {dst_idx}번 슬롯(when={target_when})로 이동했습니다.\n'
        f'아이디={removed.get("id")}\n'
        f'template={removed.get("template")}\n'
        f'현재 총 {len(data["items"])}개'
    )


TEMPLATE_HINTS = {
    'prep': '결론-이유-근거-결론(PREP) 구조',
    'pas': '문제-악화-해결(PAS) 구조',
    'listicle': '번호 리스트형(3~4포인트)',
    'whyhow': '왜(배경)-어떻게(대응) 구조',
    'brief': '짧고 담백한 브리핑형',
    'insight': '한줄 인사이트+핵심 근거형',
}


def rewrite_with_template(text, template, attempt=0):
    template = (template or '').strip().lower()
    if template not in TEMPLATE_HINTS:
        raise ValueError('unknown template')

    key = os.getenv('GEMINI_API_KEY', '').strip()
    model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite').strip()
    max_chars = int(os.getenv('DRAFT_MAX_CHARS', '250'))
    if not key:
        raise RuntimeError('GEMINI_API_KEY missing')

    retry_note = ''
    if int(attempt or 0) > 0:
        retry_note = '\n- 민감정보(키/토큰/개인식별정보/URL 쿼리 비밀값)로 오해될 수 있는 문자열 절대 금지\n- 이메일/전화번호/@긴계정/Authorization 형식 금지'

    prompt = f"""
다음 Threads 원고를 {TEMPLATE_HINTS[template]}로 재작성해줘.
조건:
- 한국어
- 해시태그 금지
- {max_chars}자 이내
- 제목 1줄 + 본문
- 과장/허위 금지{retry_note}

원문:
{text}
""".strip()

    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
    r = requests.post(
        url,
        headers={'Content-Type': 'application/json'},
        json={'generationConfig': {'temperature': 0.6}, 'contents': [{'parts': [{'text': prompt}]}]},
        timeout=60,
    )
    r.raise_for_status()
    out = r.json()['candidates'][0]['content']['parts'][0]['text']
    out = out.strip()
    out = re.sub(r'\s*#[\w가-힣_]+', '', out).strip()
    return out[: max_chars + 60]


def do_rework(arg_text):
    m = re.match(r'^(\d+)\s+([a-zA-Z_]+)$', (arg_text or '').strip())
    if not m:
        return '형식: /rework <큐번호> <템플릿>\n예: /rework 2 prep\n템플릿: prep,pas,listicle,whyhow,brief,insight'

    idx = int(m.group(1))
    template = m.group(2).lower()
    q = read_json(QUEUE, {'items': []})
    items = q.get('items', [])
    if idx < 1 or idx > len(items):
        return f'큐 번호 범위 오류: 1~{len(items)}'

    item = items[idx - 1]
    try:
        new_text = rewrite_with_template(item.get('text', ''), template, attempt=0)
        new_text = sanitize_generated_text(
            new_text,
            context=f'rework:idx={idx}:template={template}',
            regenerate_fn=lambda retry, _pattern: rewrite_with_template(item.get('text', ''), template, attempt=retry),
            max_retries=2,
            fallback_text=SAFE_REWRITE_FALLBACK,
        )
    except Exception as e:
        return f'재작성 실패: {str(e)[:180]}'

    item['text'] = new_text
    item['template'] = template
    items[idx - 1] = item
    q['items'] = items
    write_json(QUEUE, q)

    preview = new_text.replace('\n', ' ')[:120]
    return f'✅ 큐 {idx} 템플릿 변경 완료 ({template})\n미리보기: {preview}...'


def do_pick(id_csv):
    tokens = [x.strip() for x in id_csv.split(',') if x.strip()]

    rows = read_jsonl(DRAFTS)
    drafts = [r for r in rows if (r.get('status') or 'draft') == 'draft']
    if not drafts:
        return '선택 가능한 draft가 없습니다. /review 먼저 확인하세요.'

    # id 직접 선택 + 번호 선택(1-based) 모두 지원
    by_id = {str(d.get('id')): d for d in drafts}
    chosen = []
    for t in tokens:
        if t.isdigit():
            i = int(t)
            if 1 <= i <= len(drafts):
                chosen.append(drafts[i - 1])
            continue
        if t in by_id:
            chosen.append(by_id[t])

    # 중복 제거
    uniq = []
    seen = set()
    for d in chosen:
        did = str(d.get('id'))
        if did in seen:
            continue
        seen.add(did)
        uniq.append(d)
    chosen = uniq

    if not chosen:
        return '선택된 draft가 없습니다. /review 후 `/pick 1,2` 또는 `/pick <id>,<id>` 형식으로 입력하세요.'

    # 매 호출 시 최신 환경값 확인
    mix_enabled = os.getenv('MIX_CHAT_ENABLED', '1').strip() in ['1', 'true', 'True', 'yes']
    mix_every = max(1, int(os.getenv('MIX_CHAT_EVERY', '3')))

    news_items = []
    chat_items = []
    news_count = 0

    for d in chosen:
        news_items.append({
            'id': d['id'],
            'text': f"{d.get('title','')}\n\n{d.get('body','')}"
        })
        news_count += 1

        if mix_enabled and (news_count % mix_every == 0):
            prev_txt = news_items[-1]['text'] if news_items else ''
            next_txt = ''
            chat_text = make_chitchat(
                d.get('title', ''),
                [c.get('text','') for c in chat_items],
                prev_news_text=prev_txt,
                next_news_text=next_txt,
                force_question=True,
            )
            chat_text = sanitize_generated_text(
                chat_text,
                context=f'chat-pick:news_count={news_count}',
                regenerate_fn=lambda retry, _pattern: make_chitchat(
                    f"{d.get('title', '')}|pick|{news_count}|{retry}|{time.time_ns()}",
                    [c.get('text','') for c in chat_items],
                    prev_news_text=prev_txt,
                    next_news_text=next_txt,
                    force_question=True,
                ),
                max_retries=2,
                fallback_text=SAFE_CHAT_FALLBACK,
            )
            chat_items.append({
                'id': f"chat_{int(time.time())}_{news_count}",
                'text': chat_text,
                'template': 'chat',
            })

    new_batch = merge_with_chat_schedule(news_items, chat_items, start_from=datetime.datetime.now())
    chosen_ids = {str(d.get('id')) for d in chosen}

    stats = {'added_news': 0, 'added_chat': 0, 'queue_total': 0}

    def _queue_mutator(q):
        existing = q.get('items', [])
        existing_ids = {str(x.get('id')) for x in existing}

        filtered_batch = []
        for it in new_batch:
            iid = str(it.get('id'))
            if iid in existing_ids:
                continue
            filtered_batch.append(it)
            existing_ids.add(iid)

        if existing and filtered_batch:
            last_when = max(x.get('when', '') for x in existing)
            try:
                last_dt = datetime.datetime.strptime(last_when, '%Y-%m-%d %H:%M')
                first_new_dt = datetime.datetime.strptime(filtered_batch[0]['when'], '%Y-%m-%d %H:%M')
                delta = (last_dt + datetime.timedelta(hours=2)) - first_new_dt
                for it in filtered_batch:
                    dt = datetime.datetime.strptime(it['when'], '%Y-%m-%d %H:%M') + delta
                    it['when'] = dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                pass

        q['items'] = existing + filtered_batch
        # 큐 재생성/추가 시에는 채팅 삽입 위치 커서를 리셋
        q['__chat_insert_cursor'] = 0
        stats['added_news'] = sum(1 for it in filtered_batch if str(it.get('id')) in chosen_ids)
        stats['added_chat'] = sum(1 for it in filtered_batch if str(it.get('template') or '') == 'chat')
        stats['queue_total'] = len(q['items'])
        return q

    update_json_locked(QUEUE, {'items': []}, _queue_mutator)

    # /review에서 재노출되지 않도록 선택된 draft 상태를 queued로 전환
    changed = 0
    for r in rows:
        if str(r.get('id')) in chosen_ids and (r.get('status') or 'draft') == 'draft':
            r['status'] = 'queued'
            changed += 1
    if changed:
        with open(DRAFTS, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    global MIX_AUTO_OFF_REMAIN
    if MIX_AUTO_OFF_REMAIN > 0 and stats['added_chat'] > 0:
        MIX_AUTO_OFF_REMAIN = max(0, MIX_AUTO_OFF_REMAIN - stats['added_chat'])
        if MIX_AUTO_OFF_REMAIN <= 0:
            set_mix_mode(False)
            tail = '\n(지정한 횟수 배치 완료 후 mix_off 처리했습니다.)'
        else:
            tail = f'\n(남은 자동 삽입: {MIX_AUTO_OFF_REMAIN}개)'
    else:
        tail = ''

    if stats['added_news'] == 0 and stats['added_chat'] == 0:
        return f'중복으로 추가된 항목이 없습니다. 현재 큐: {stats["queue_total"]}개' + tail
    return f'예약큐 추가 완료: 뉴스 {stats["added_news"]}개 + 잡담 {stats["added_chat"]}개\n현재 큐: {stats["queue_total"]}개' + tail


def show_input_menu():
    send(
        "명령 메뉴\n"
        "1) 큐 보기 (/schedule)\n"
        "2) 검수 보기 (/review)\n"
        "3) 큐 항목 템플릿 변경 시작\n\n"
        "숫자만 입력하세요 (예: 3)"
    )


def get_updates():
    global last_update_id, ui_state
    r = requests.get(f'https://api.telegram.org/bot{TOKEN}/getUpdates', params={'offset': last_update_id + 1, 'timeout': 10}, timeout=15)
    data = r.json()
    if not data.get('ok'):
        return
    for u in data.get('result', []):
        last_update_id = max(last_update_id, u['update_id'])
        m = u.get('message', {})
        cid = str(m.get('chat', {}).get('id'))
        text = (m.get('text') or '').strip()
        if cid != CHAT_ID:
            continue

        # 단계형 입력 UI 처리
        if text in ['/입력', '/input']:
            ui_state = {'mode': None, 'queue_idx': None}
            show_input_menu()
            continue

        if ui_state.get('mode') == 'pick_queue_item' and text.isdigit():
            idx = int(text)
            q = read_json(QUEUE, {'items': []}).get('items', [])
            if 1 <= idx <= len(q):
                ui_state['mode'] = 'pick_template'
                ui_state['queue_idx'] = idx
                send(
                    f"큐 {idx} 선택됨. 템플릿 번호를 입력하세요:\n"
                    "1) PREP\n2) PAS\n3) LISTICLE\n4) WHYHOW\n5) BRIEF\n6) INSIGHT"
                )
            else:
                send(f"큐 번호 범위 오류: 1~{len(q)}")
            continue

        if ui_state.get('mode') == 'pick_template' and text.isdigit():
            m = {'1':'prep','2':'pas','3':'listicle','4':'whyhow','5':'brief','6':'insight'}
            t = m.get(text)
            if not t:
                send('템플릿 번호는 1~6만 가능합니다.')
                continue
            idx = ui_state.get('queue_idx')
            send(do_rework(f"{idx} {t}"))
            ui_state = {'mode': None, 'queue_idx': None}
            continue

        # 메뉴 숫자 처리
        if ui_state.get('mode') is None and text in ['1','2','3']:
            if text == '1':
                text = '/schedule'
            elif text == '2':
                text = '/review'
            elif text == '3':
                q = read_json(QUEUE, {'items': []}).get('items', [])
                if not q:
                    send('예약큐가 비어있습니다.')
                else:
                    ui_state['mode'] = 'pick_queue_item'
                    lines = ['템플릿 변경할 큐 번호를 입력하세요:']
                    for n, i in enumerate(q[:20], 1):
                        lines.append(f"{n}) {i.get('id')} @ {i.get('when')}")
                    send('\n'.join(lines))
                continue

        if text in ['/collect_extract', '/collect_news', '/extract_news', '뉴스 수집']:
            def _cb(code, out):
                send(f'collect_extract exit={code}\\n{out[-600:]}')
            ok, msg = run_async_job(
                'collect_extract',
                JOB_DEFINITIONS['collect_extract']['cmd'],
                success_cb=_cb,
            )
            send(msg)
        elif text in ['/generate_news', '뉴스생성', '뉴스 생성']:
            old = archive_pending_drafts()
            def _cb(code, out):
                send(f'generate_news exit={code} (archived_old={old})\n{out[-400:]}')
            ok, msg = run_async_job(
                'generate_news',
                JOB_DEFINITIONS['generate_news']['cmd'],
                success_cb=_cb,
            )
            send(msg)
        elif text in ['/review', '뉴스 리뷰']:
            d = pending_drafts()[:20]
            if not d:
                send('검수할 draft가 없습니다.')
            else:
                hm = hot_map_by_url()
                lines = [f'검수 대기 {len(d)}개']
                for i, x in enumerate(d, 1):
                    fmt = (x.get('format') or '-').upper()
                    preview = (x.get('body','') or '').replace('\n', ' ')[:90]
                    h = hm.get(str(x.get('url') or ''), {})
                    score = h.get('score')
                    bd = h.get('score_breakdown', {})
                    score_txt = f"hot={score}" if score is not None else 'hot=-'
                    bd_txt = ''
                    if bd:
                        bd_txt = f" (f{bd.get('freshness',0)}/b{bd.get('buzz',0)}/s{bd.get('source',0)}/r{bd.get('relevance',0)})"
                    lines.append(
                        f"{i}) [{fmt}] {x.get('title','')[:28]}\n"
                        f"   id: {x['id']} | {score_txt}{bd_txt}\n"
                        f"   미리보기: {preview}..."
                    )
                lines.append('\n선택: /pick 1,2  (번호 선택 가능)')
                lines.append('상세보기: /show 1(또는 /showd 1)은 검수목록, /showq 1은 스케줄큐를 봅니다. 전체 요약은 /showq all')
                send('\n'.join(lines))
        elif text.startswith('/showd ') or text.startswith('/show '):
            key = text.split(' ', 1)[1].strip()

            d = pending_drafts()
            q = read_json(QUEUE, {'items': []}).get('items', [])

            # 먼저 draft 기준으로 해석(검수리스트와 번호가 일치하는 UX 우선)
            target = None
            target_is_queue = False
            if key.isdigit():
                i = int(key)
                if 1 <= i <= len(d):
                    target = d[i - 1]
                elif 1 <= i <= len(q):
                    target = q[i - 1]
                    target_is_queue = True
            else:
                for x in d:
                    if str(x.get('id')) == key:
                        target = x
                        break
                if not target:
                    for x in q:
                        if str(x.get('id')) == key:
                            target = x
                            target_is_queue = True
                            break

            if not target:
                send('대상을 찾지 못했습니다. /show <번호 또는 draft_id> 또는 /showq <큐번호>로 입력하세요.')
            elif target_is_queue:
                t = (target.get('template') or '-').upper()
                idx = '?'
                try:
                    idx = q.index(target) + 1
                except Exception:
                    pass
                send(
                    f"[QUEUE {idx}] [{t}]\n"
                    f"id: {target.get('id')}\n"
                    f"when: {target.get('when')}\n\n"
                    f"{target.get('text','')}"
                )
            else:
                fmt = (target.get('format') or '-').upper()
                send(
                    f"[DRAFT] [{fmt}] {target.get('title','')}\n"
                    f"id: {target.get('id')}\n"
                    f"url: {target.get('url','')}\n\n"
                    f"{target.get('body','')}"
                )
        elif text.startswith('/showq'):
            parts = text.split(' ', 1)
            key = parts[1].strip() if len(parts) > 1 else ''
            q = read_json(QUEUE, {'items': []}).get('items', [])

            if key == '' or key == 'all':
                if not q:
                    send('예약큐가 비어있습니다.')
                else:
                    msg = ['[예약큐 요약]']
                    for n, it in enumerate(q, 1):
                        t = (it.get('template') or 'news').upper()
                        title = it.get('title') or (it.get('text') or '')[:28]
                        when = it.get('when', '-')
                        msg.append(f"{n}) [{t}] {when} | {title[:60]}")
                    msg.append('상세: /showq <번호 or id>')
                    send('\n'.join(msg))
                
            elif key.isdigit():
                idx = int(key)
                if 1 <= idx <= len(q):
                    item = q[idx - 1]
                    t = (item.get('template') or '-').upper()
                    send(
                        f"[QUEUE {idx}] [{t}]\n"
                        f"id: {item.get('id')}\n"
                        f"when: {item.get('when')}\n\n"
                        f"{item.get('text','')}"
                    )
                else:
                    send(f'큐 번호 범위 오류: 1~{len(q)}')
            else:
                for item in q:
                    if str(item.get('id')) == key:
                        t = (item.get('template') or '-').upper()
                        send(
                            f"[QUEUE] [{t}]\n"
                            f"id: {item.get('id')}\n"
                            f"when: {item.get('when')}\n\n"
                            f"{item.get('text','')}"
                        )
                        break
                else:
                    send('큐 항목을 찾지 못했습니다. /showq <큐번호/큐id>')
        elif text.startswith('/pick '):
            send(do_pick(text.split(' ', 1)[1]))
        elif text in ['/schedule', '스케쥴관리']:
            q = read_json(QUEUE, {'items': []}).get('items', [])
            if not q:
                send('예약큐 비어있음')
            else:
                msg = ['예약큐']
                for n, i in enumerate(q[:20], 1):
                    t = (i.get('template') or 'news').upper()
                    msg.append(f"{n}) [{t}] {i['id']} @ {i['when']}")
                msg.append('템플릿 변경: /rework 2 prep')
                send('\n'.join(msg))
        elif text == '/rework':
            q = read_json(QUEUE, {'items': []}).get('items', [])
            if not q:
                send('예약큐가 비어있습니다.')
            else:
                lines = ['템플릿 변경할 큐 번호를 입력하세요:']
                for n, i in enumerate(q[:20], 1):
                    lines.append(f"{n}) {i.get('id')} @ {i.get('when')}")
                lines.append('\n예: /rework 1')
                send('\n'.join(lines))
        elif text.startswith('/rework '):
            arg = text.split(' ', 1)[1].strip()
            # /rework 1 처럼 큐 번호만 오면 템플릿 선택 단계로 전환
            if arg.isdigit():
                idx = int(arg)
                q = read_json(QUEUE, {'items': []}).get('items', [])
                if 1 <= idx <= len(q):
                    ui_state['mode'] = 'pick_template'
                    ui_state['queue_idx'] = idx
                    send(
                        f"큐 {idx} 선택됨. 템플릿 번호를 입력하세요:\n"
                        "1) PREP\n2) PAS\n3) LISTICLE\n4) WHYHOW\n5) BRIEF\n6) INSIGHT"
                    )
                else:
                    send(f"큐 번호 범위 오류: 1~{len(q)}")
            else:
                # /rework 1 prep 기존 형식도 계속 지원
                send(do_rework(arg))
        elif text.startswith('/publish_now '):
            # 수동 발행도 news2 전용 CDP/프로필 가드 강제
            run('systemctl --user start news2-cdp.service >/dev/null 2>&1 || true')
            c0, out0 = run('python3 /home/ubuntu/threads-bot-news2/ensure_profile.py')
            if c0 != 0:
                send('❌ 발행 차단: news2 프로필(fichte_news) CDP가 아닙니다.\n' + out0[-300:])
            else:
                did = text.split(' ', 1)[1].strip()
                drafts = {str(x.get('id')): x for x in pending_drafts()}
                d = drafts.get(did)
                if not d:
                    send('해당 draft id를 찾지 못했습니다.')
                else:
                    publish_js = os.getenv('THREADS_PUBLISH_JS', '/home/ubuntu/threads-bot-news2/publish_news2.js')
                    news_handle = os.getenv('NEWS_THREADS_OWN_HANDLE', '').replace('@', '').strip()
                    payload = f"{d.get('title','')}\n\n{d.get('body','')}"
                    cdp_url = os.getenv('NEWS_THREADS_CDP_URL', 'http://127.0.0.1:9223/')
                    c, out = run_publish_news2(publish_js, payload, own_handle=news_handle, cdp_url=cdp_url)
                    send(f'publish_now exit={c}\n{out[-500:]}')
        elif text.startswith('/mix_on'):
            parts = text.split()
            auto_off = 0
            if len(parts) > 1:
                try:
                    auto_off = max(1, int(parts[1]))
                except Exception:
                    auto_off = 0
            set_mix_mode(True)
            global MIX_AUTO_OFF_REMAIN
            MIX_AUTO_OFF_REMAIN = auto_off
            if auto_off > 0:
                send(f'✅ 잡담 자동 삽입 ON (남은 {auto_off}개 후 자동 OFF)')
            else:
                send('✅ 잡담 삽입 ON')
        elif text in ['/mix_off', 'mix_off', '/mixoff', 'mixoff']:
            set_mix_mode(False)
            MIX_AUTO_OFF_REMAIN = 0
            send('⏸ 잡담 삽입 OFF')
        elif text in ['mix_on', 'mix_off', 'mixoff']:
            # 기존 short alias 호환
            if text == 'mix_on':
                set_mix_mode(True)
                MIX_AUTO_OFF_REMAIN = 0
                send('✅ 잡담 삽입 ON')
            elif text in ['mix_off', 'mixoff']:
                set_mix_mode(False)
                MIX_AUTO_OFF_REMAIN = 0
                send('⏸ 잡담 삽입 OFF')
        elif text.startswith('/chat') or text.startswith('잡담'):
            arg = text.split(' ', 1)[1] if ' ' in text else ''
            if arg and arg.strip().isdigit():
                c = int(arg.strip())
            else:
                c = 1
            send(insert_chats_into_queue(c))
        elif text.startswith('/queue_move '):
            body = text[len('/queue_move '):].strip()
            # 허용 형식: /queue_move <src_idx> <dst_idx>  또는 'src>dst'
            parts = [p for p in re.split(r'\s+', body) if p]
            if len(parts) < 2 and '>' in body:
                parts = [x for x in body.replace('＞', '>').split('>') if x.strip()]

            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                src = int(parts[0])
                dst = int(parts[1])
                send(move_queue_item(src, dst))
            else:
                send('형식: /queue_move <원래번호> <목표번호>\n예: /queue_move 5 8')
        elif text in ['/status', '상태']:
            links = len(read_jsonl(DATA / 'news_links.jsonl'))
            arts = len(read_jsonl(DATA / 'articles.jsonl'))
            drafts = len(pending_drafts())
            queue = len(read_json(QUEUE, {'items': []}).get('items', []))
            mix_on = os.getenv('MIX_CHAT_ENABLED', '1')
            mix_every = os.getenv('MIX_CHAT_EVERY', '3')
            pending_auto_off = MIX_AUTO_OFF_REMAIN if 'MIX_AUTO_OFF_REMAIN' in globals() else 0

            job_lines = []
            for key, meta in background_jobs.items():
                state = meta.get('status', 'unknown')
                if state == 'running':
                    job_lines.append(f"- {meta.get('label', key)}: 진행중")
                elif state == 'done':
                    job_lines.append(f"- {meta.get('label', key)}: 완료 code={meta.get('exit_code')}")
                else:
                    job_lines.append(f"- {meta.get('label', key)}: {state}")
            jobs = ('\n' + '\n'.join(job_lines)) if job_lines else '\n(백그라운드 작업 없음)'

            send(f'status\nlinks={links}\narticles={arts}\ndrafts(pending)={drafts}\nqueue={queue}\nmix_chat={mix_on} (every {mix_every})\nmix_auto_off={pending_auto_off}\njobs={jobs}')


def main():
    if not TOKEN or not CHAT_ID:
        print('TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID required')
        return
    send('news2 watcher 시작')
    while True:
        try:
            get_updates()
            time.sleep(1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            send(f'watcher error: {str(e)[:200]}')
            time.sleep(3)


if __name__ == '__main__':
    main()
