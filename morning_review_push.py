#!/usr/bin/env python3
import os, json, requests
from dotenv import dotenv_values

cfg = dotenv_values('/home/ubuntu/threads-bot-news2/.env')
TOKEN = (cfg.get('TELEGRAM_BOT_TOKEN') or '').strip()
CHAT_ID = (cfg.get('TELEGRAM_CHAT_ID') or '').strip()
DRAFTS = '/home/ubuntu/threads-bot-news2/data/drafts.jsonl'

if not TOKEN or not CHAT_ID:
    raise SystemExit('missing telegram env')

rows = []
if os.path.exists(DRAFTS):
    with open(DRAFTS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get('status') or 'draft') == 'draft':
                rows.append(r)

msg = [f'🗞 아침 검수 알림 (draft {len(rows)}개)']
for i, r in enumerate(rows[:10], 1):
    fmt = (r.get('format') or '-').upper()
    title = (r.get('title') or '')[:40]
    msg.append(f"{i}) [{fmt}] {title}")
msg.append('\n검수: /review\n선택: /pick 1,2\n일정확인: /schedule')

requests.post(
    f'https://api.telegram.org/bot{TOKEN}/sendMessage',
    json={'chat_id': CHAT_ID, 'text': '\n'.join(msg)},
    timeout=10,
)
print('PUSH_OK')
