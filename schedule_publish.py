#!/usr/bin/env python3
import os, subprocess, datetime
from pathlib import Path
from common import DATA, read_json, write_json, update_json_locked

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
    else:
        print('PUBLISH_ERR', r.stderr[-300:])


if __name__ == '__main__':
    main()
