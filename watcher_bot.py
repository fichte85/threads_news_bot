#!/usr/bin/env python3
import os, time, datetime, json, requests, subprocess, re
from common import DATA, read_jsonl, read_json, write_json

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


def send(text):
    if not TOKEN or not CHAT_ID:
        return
    keyboard = {
        'keyboard': [
            [{'text': '/collect_extract'}, {'text': '/generate_news'}],
            [{'text': '/review'}, {'text': '/schedule'}, {'text': '/status'}],
            [{'text': '/mix_on'}, {'text': '/mix_off'}],
        ],
        'resize_keyboard': True,
        'one_time_keyboard': False,
    }
    requests.post(
        f'https://api.telegram.org/bot{TOKEN}/sendMessage',
        json={'chat_id': CHAT_ID, 'text': text, 'reply_markup': json.dumps(keyboard, ensure_ascii=False)}, timeout=10
    )


def run(cmd):
    r = subprocess.run(['bash', '-lc', cmd], cwd='/home/ubuntu/threads-bot-news2', capture_output=True, text=True)
    return r.returncode, (r.stdout + '\n' + r.stderr)[-1000:]


def pending_drafts():
    rows = read_jsonl(DRAFTS)
    return [r for r in rows if (r.get('status') or 'draft') == 'draft']


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


def build_news_slots(n):
    # 뉴스는 2시간 간격 고정(정규 슬롯)
    start = os.getenv('PUBLISH_START', '10:30')
    sh, sm = map(int, start.split(':'))
    day = datetime.date.today()
    out = []
    for i in range(n):
        out.append(datetime.datetime.combine(day, datetime.time(sh, sm)) + datetime.timedelta(hours=2 * i))
    return out


def merge_with_chat_schedule(news_items, chat_items):
    # news_items/chat_items: [{'id','text','template?'}]
    news_slots = build_news_slots(len(news_items))
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


def make_chitchat(seed_text=''):
    pool = [
        '요즘은 빠르게 만들고 작게 검증하는 루틴이 제일 효율적이네요. 오늘도 작은 실험 하나 추가합니다.',
        '뉴스를 많이 보는 것보다, 내 일에 바로 연결되는 한 줄 인사이트를 남기는 게 더 중요하더라고요.',
        '자동화는 화려함보다 안정성이 먼저인 것 같아요. 천천히 굴려도 끊기지 않는 게 결국 이깁니다.',
        '오늘도 기록 하나 남깁니다. 방향이 맞으면 속도는 나중에 자연스럽게 붙더라고요.',
        'AI는 결국 도구고, 결과를 만드는 건 운영 루틴이더라고요. 오늘도 실행 기준으로 가봅니다.',
    ]
    idx = abs(hash(seed_text or str(datetime.datetime.now()))) % len(pool)
    return pool[idx]


TEMPLATE_HINTS = {
    'prep': '결론-이유-근거-결론(PREP) 구조',
    'pas': '문제-악화-해결(PAS) 구조',
    'listicle': '번호 리스트형(3~4포인트)',
    'whyhow': '왜(배경)-어떻게(대응) 구조',
    'brief': '짧고 담백한 브리핑형',
    'insight': '한줄 인사이트+핵심 근거형',
}


def rewrite_with_template(text, template):
    template = (template or '').strip().lower()
    if template not in TEMPLATE_HINTS:
        raise ValueError('unknown template')

    key = os.getenv('GEMINI_API_KEY', '').strip()
    model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite').strip()
    max_chars = int(os.getenv('DRAFT_MAX_CHARS', '250'))
    if not key:
        raise RuntimeError('GEMINI_API_KEY missing')

    prompt = f"""
다음 Threads 원고를 {TEMPLATE_HINTS[template]}로 재작성해줘.
조건:
- 한국어
- 해시태그 금지
- {max_chars}자 이내
- 제목 1줄 + 본문
- 과장/허위 금지

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
        new_text = rewrite_with_template(item.get('text', ''), template)
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
    drafts = pending_drafts()
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

    q = read_json(QUEUE, {'items': []})

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
            chat_items.append({
                'id': f"chat_{int(time.time())}_{news_count}",
                'text': make_chitchat(d.get('title', '')),
                'template': 'chat',
            })

    new_batch = merge_with_chat_schedule(news_items, chat_items)

    # 기존 큐가 있으면 뒤에 이어붙이되, 새 배치는 마지막 예약 이후부터 2시간 단위로 이동
    existing = q.get('items', [])
    if existing:
        last_when = max(x.get('when', '') for x in existing)
        try:
            last_dt = datetime.datetime.strptime(last_when, '%Y-%m-%d %H:%M')
            first_new_dt = datetime.datetime.strptime(new_batch[0]['when'], '%Y-%m-%d %H:%M')
            delta = (last_dt + datetime.timedelta(hours=2)) - first_new_dt
            for it in new_batch:
                dt = datetime.datetime.strptime(it['when'], '%Y-%m-%d %H:%M') + delta
                it['when'] = dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass

    q['items'] = existing + new_batch
    write_json(QUEUE, q)
    return f'예약큐 추가 완료: 뉴스 {len(news_items)}개 + 잡담 {len(chat_items)}개\n현재 큐: {len(q["items"])}개'


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

        if text in ['/collect_extract', '/collect_news', '/extract_news']:
            c, out = run('python3 rss_collect.py && python3 extract_articles.py')
            send(f'collect_extract exit={c}\n{out[-500:]}')
        elif text == '/generate_news':
            old = archive_pending_drafts()
            c, out = run('python3 hot_only.py && python3 generate_drafts.py')
            send(f'generate_news exit={c} (archived_old={old})\n{out[-400:]}')
        elif text == '/review':
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
                lines.append('상세보기: /show 1  또는 /show <id>')
                send('\n'.join(lines))
        elif text.startswith('/show '):
            key = text.split(' ', 1)[1].strip()

            # 숫자 입력이면 예약큐 기준으로 먼저 보여줌(/schedule과 번호 일치)
            if key.isdigit():
                idx = int(key)
                q = read_json(QUEUE, {'items': []}).get('items', [])
                if 1 <= idx <= len(q):
                    item = q[idx - 1]
                    t = (item.get('template') or '-').upper()
                    send(
                        f"[QUEUE {idx}] [{t}]\n"
                        f"id: {item.get('id')}\n"
                        f"when: {item.get('when')}\n\n"
                        f"{item.get('text','')}"
                    )
                    continue

            # 그 외는 draft 기준(id 또는 번호)
            d = pending_drafts()
            target = None
            if key.isdigit():
                i = int(key)
                if 1 <= i <= len(d):
                    target = d[i - 1]
            else:
                for x in d:
                    if str(x.get('id')) == key:
                        target = x
                        break
            if not target:
                send('대상을 찾지 못했습니다. /show <큐번호> 또는 /show <draft_id> 로 입력하세요.')
            else:
                fmt = (target.get('format') or '-').upper()
                send(
                    f"[DRAFT] [{fmt}] {target.get('title','')}\n"
                    f"id: {target.get('id')}\n"
                    f"url: {target.get('url','')}\n\n"
                    f"{target.get('body','')}"
                )
        elif text.startswith('/pick '):
            send(do_pick(text.split(' ', 1)[1]))
        elif text == '/schedule':
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
                    if news_handle:
                        cmd = f"THREADS_CDP_URL={cdp_url} THREADS_OWN_HANDLE={news_handle} node {publish_js} {repr(payload)}"
                    else:
                        cmd = f"THREADS_CDP_URL={cdp_url} node {publish_js} {repr(payload)}"
                    c, out = run(cmd)
                    send(f'publish_now exit={c}\n{out[-500:]}')
        elif text in ['/mix_on', 'mix_on']:
            run("python3 - <<'PY'\nfrom pathlib import Path\np=Path('/home/ubuntu/threads-bot-news2/.env')\ns=p.read_text(encoding='utf-8')\nif 'MIX_CHAT_ENABLED=' in s:\n s='\\n'.join(['MIX_CHAT_ENABLED=1' if ln.startswith('MIX_CHAT_ENABLED=') else ln for ln in s.splitlines()])\nelse:\n s+='\\nMIX_CHAT_ENABLED=1\\n'\np.write_text(s,encoding='utf-8')\nprint('ok')\nPY")
            send('✅ 잡담 삽입 ON')
        elif text in ['/mix_off', 'mix_off']:
            run("python3 - <<'PY'\nfrom pathlib import Path\np=Path('/home/ubuntu/threads-bot-news2/.env')\ns=p.read_text(encoding='utf-8')\nif 'MIX_CHAT_ENABLED=' in s:\n s='\\n'.join(['MIX_CHAT_ENABLED=0' if ln.startswith('MIX_CHAT_ENABLED=') else ln for ln in s.splitlines()])\nelse:\n s+='\\nMIX_CHAT_ENABLED=0\\n'\np.write_text(s,encoding='utf-8')\nprint('ok')\nPY")
            send('⏸ 잡담 삽입 OFF')
        elif text == '/status':
            links = len(read_jsonl(DATA / 'news_links.jsonl'))
            arts = len(read_jsonl(DATA / 'articles.jsonl'))
            drafts = len(pending_drafts())
            queue = len(read_json(QUEUE, {'items': []}).get('items', []))
            mix_on = os.getenv('MIX_CHAT_ENABLED', '1')
            mix_every = os.getenv('MIX_CHAT_EVERY', '3')
            send(f'status\nlinks={links}\narticles={arts}\ndrafts(pending)={drafts}\nqueue={queue}\nmix_chat={mix_on} (every {mix_every})')


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
