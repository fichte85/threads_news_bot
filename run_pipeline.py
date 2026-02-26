#!/usr/bin/env python3
import subprocess

steps = [
    'python3 rss_collect.py',
    'python3 extract_articles.py',
    'python3 hot_only.py',
    'python3 generate_drafts.py',
]

for s in steps:
    print('RUN', s)
    r = subprocess.run(['bash', '-lc', s], cwd='/home/ubuntu/threads-bot-news2')
    if r.returncode != 0:
        print('STOP on error', s)
        break
