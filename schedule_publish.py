#!/usr/bin/env python3
import os, subprocess, datetime, json
import urllib.request
from pathlib import Path
from common import DATA, read_json, write_json, update_json_locked

QUEUE = DATA / 'publish_queue.json'
STATE = DATA / 'publish_state.json'


def send_telegram(msg: str) -> bool:
    token = (os.getenv('TELEGRAM_BOT_TOKEN') or '').strip()
    chat_id = (os.getenv('TELEGRAM_CHAT_ID') or '').strip()
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': msg,
        'disable_web_page_preview': True,
    }

    # 최소 의존성 우선: urllib 사용, 실패 시 requests fallback
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json; charset=utf-8'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            if 200 <= int(getattr(resp, 'status', 0) or 0) < 300:
                return True
    except Exception as e:
        print(f'TG_NOTIFY_WARN urllib: {str(e)[:180]}')

    try:
        import requests  # type: ignore
        rr = requests.post(url, json=payload, timeout=8)
        if rr.ok:
            return True
        print(f'TG_NOTIFY_WARN requests: HTTP {rr.status_code} {rr.text[:180]}')
    except Exception as e:
        print(f'TG_NOTIFY_WARN requests: {str(e)[:180]}')

    return False


def _fmt_ok_msg(head: dict, now: datetime.datetime) -> str:
    lines = [
        '✅ [뉴스봇2 발행] OK',
        f"- id: {head.get('id', '-')}",
        f"- when: {head.get('when') or now.strftime('%Y-%m-%d %H:%M')}",
    ]
    title = (head.get('title') or '').strip()
    if title:
        lines.append(f'- title: {title}')
    url = (head.get('url') or '').strip()
    if url:
        lines.append(f'- url: {url}')
    return '\n'.join(lines)


def _fmt_err_msg(head: dict, now: datetime.datetime, stderr_text: str) -> str:
    err = (stderr_text or '').strip().replace('\n', ' ')
    err = err[:280] if err else '-'
    return '\n'.join([
        '❌ [뉴스봇2 발행] ERR',
        f"- id: {head.get('id', '-')}",
        f"- when: {head.get('when') or now.strftime('%Y-%m-%d %H:%M')}",
        f'- stderr: {err}',
    ])


def main():
    # news2 전용 CDP(9223) 기동 후 프로필 검증
    subprocess.run(['bash', '-lc', 'systemctl --user start news2-cdp.service >/dev/null 2>&1 || true'])
    chk = subprocess.run(['python3', '/home/ubuntu/threads-bot-news2/ensure_profile.py'], capture_output=True, text=True)
    print((chk.stdout or '').strip())
    if chk.returncode != 0:
        print('ABORT_PUBLISH_BY_PROFILE_GUARD')
        return

    q = read_json(QUEUE, {'items': []})
    state = read_json(STATE, {'date': '', 'count': 0})
    items = q.get('items', [])
    if not items:
        print('QUEUE_EMPTY')
        return

    now = datetime.datetime.now()
    today = str(now.date())
    # 한도 정책: 0 또는 음수면 무제한 모드
    env_limit = int(os.getenv('DAILY_PUBLISH_LIMIT', '11').strip() or 0)
    if env_limit <= 0:
        limit = 10**9
    else:
        limit = env_limit

    if state.get('date') != today:
        state = {'date': today, 'count': 0}

    if state.get('count', 0) >= limit:
        print('DAILY_LIMIT_REACHED', state['count'], '/', limit)
        write_json(STATE, state)
        return

    head = items[0]
    when = datetime.datetime.strptime(head['when'], '%Y-%m-%d %H:%M')
    if now < when:
        print('NOT_YET', head['when'])
        return

    publish_js = os.getenv('THREADS_PUBLISH_JS', '/home/ubuntu/threads-bot-news2/publish_news2.js')
    news_handle = os.getenv('NEWS_THREADS_OWN_HANDLE', '').replace('@', '').strip()
    draft = head.get('text', '').strip()
    if not draft:
        def _pop_empty(cur):
            cur_items = cur.get('items', [])
            if cur_items and str(cur_items[0].get('id')) == str(head.get('id')):
                cur['items'] = cur_items[1:]
            return cur

        update_json_locked(QUEUE, {'items': []}, _pop_empty)
        print('SKIP_EMPTY_DRAFT')
        return

    cmd = ['node', publish_js, draft]
    env = os.environ.copy()
    env['THREADS_CDP_URL'] = os.getenv('NEWS_THREADS_CDP_URL', 'http://127.0.0.1:9223/')
    if news_handle:
        env['THREADS_OWN_HANDLE'] = news_handle
    # node_modules fallback: news2 작업공간 자체에 node_modules가 없어도 bot1 모듈 공유 경로를 사용
    nm_candidates = [
        '/home/ubuntu/threads-bot/node_modules',
        '/home/ubuntu/threads-bot-news2/node_modules',
    ]
    node_path = env.get('NODE_PATH', '')
    for np in nm_candidates:
        if Path(np).exists():
            if node_path:
                node_path = f"{np}:{node_path}"
            else:
                node_path = np
            break
    if node_path:
        env['NODE_PATH'] = node_path

    # 실행 경로를 스크립트 위치로 맞춰 모듈 탐색 기준 일치화
    env['THREADS_PUBLISH_LOG'] = '/home/ubuntu/threads-bot-news2/threads_publish_full.log'
    r = subprocess.run(cmd, cwd='/home/ubuntu/threads-bot-news2', env=env, capture_output=True, text=True)
    print(r.stdout[-400:])
    if r.returncode == 0:
        def _pop_published(cur):
            cur_items = cur.get('items', [])
            if cur_items and str(cur_items[0].get('id')) == str(head.get('id')):
                cur['items'] = cur_items[1:]
            return cur

        update_json_locked(QUEUE, {'items': []}, _pop_published)
        state['count'] = int(state.get('count', 0)) + 1
        write_json(STATE, state)
        print('PUBLISH_OK', head.get('id'))
        try:
            send_telegram(_fmt_ok_msg(head, now))
        except Exception as e:
            print(f'TG_NOTIFY_WARN ok: {str(e)[:180]}')
    else:
        err_tail = (r.stderr or '')[-300:]
        print('PUBLISH_ERR', err_tail)
        try:
            send_telegram(_fmt_err_msg(head, now, err_tail))
        except Exception as e:
            print(f'TG_NOTIFY_WARN err: {str(e)[:180]}')


if __name__ == '__main__':
    main()
