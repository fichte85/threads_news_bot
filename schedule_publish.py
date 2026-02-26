#!/usr/bin/env python3
import os, subprocess, datetime
from common import DATA, read_json, write_json

QUEUE = DATA / 'publish_queue.json'
STATE = DATA / 'publish_state.json'


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
    limit = int(os.getenv('DAILY_PUBLISH_LIMIT', '2'))

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
        q['items'] = items[1:]
        write_json(QUEUE, q)
        print('SKIP_EMPTY_DRAFT')
        return

    cmd = ['node', publish_js, draft]
    env = os.environ.copy()
    env['THREADS_CDP_URL'] = os.getenv('NEWS_THREADS_CDP_URL', 'http://127.0.0.1:9223/')
    if news_handle:
        env['THREADS_OWN_HANDLE'] = news_handle
    r = subprocess.run(cmd, cwd='/home/ubuntu/threads-bot', env=env, capture_output=True, text=True)
    print(r.stdout[-400:])
    if r.returncode == 0:
        q['items'] = items[1:]
        write_json(QUEUE, q)
        state['count'] = int(state.get('count', 0)) + 1
        write_json(STATE, state)
        print('PUBLISH_OK', head.get('id'))
    else:
        print('PUBLISH_ERR', r.stderr[-300:])


if __name__ == '__main__':
    main()
