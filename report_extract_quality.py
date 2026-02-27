#!/usr/bin/env python3
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE = Path('/home/ubuntu/threads-bot-news2')
DATA = BASE / 'data'
PROCESSED = DATA / 'processed_news_links.jsonl'
DEBUG = DATA / 'extract_debug.jsonl'
REPORT_DIR = DATA / 'reports'

KST = timezone(timedelta(hours=9))


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _parse_ts(v):
    if not v:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        if s.endswith('Z'):
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST)
    except Exception:
        return None


def _safe_pct(n, d):
    return 0.0 if d <= 0 else (n / d * 100.0)


def _target_days(days: int):
    today = datetime.now(KST).date()
    # days=1 -> 오늘 1일치, days=2 -> 어제+오늘
    return [today - timedelta(days=i) for i in range(days - 1, -1, -1)]


def _collect_processed(days_set):
    by_day = {d: [] for d in days_set}
    for row in _iter_jsonl(PROCESSED):
        dt = _parse_ts(row.get('ts'))
        if not dt:
            continue
        day = dt.date()
        if day in by_day:
            by_day[day].append(row)
    return by_day


def _collect_debug_index(days_set):
    # (day, url) -> debug row (latest)
    idx = {}
    for row in _iter_jsonl(DEBUG):
        dt = _parse_ts(row.get('ts'))
        if not dt:
            continue
        day = dt.date()
        if day not in days_set:
            continue
        url = row.get('url')
        if not url:
            continue
        idx[(day, url)] = row
    return idx


def _error_type(row):
    msg = str(row.get('error', '') or '').lower().strip()
    if not msg:
        return 'error_unknown'
    for k in [
        'timeout', 'timed out',
        '403', '404', '429', '500', '502', '503', '504',
        'ssl', 'connection', 'max retries', 'name or service not known',
    ]:
        if k in msg:
            return k
    return msg[:60]


def _domain_from_row(row, debug_row=None):
    for key in ['final_url', 'resolved_url', 'url']:
        u = row.get(key)
        if u:
            try:
                host = urlparse(str(u)).netloc
                if host:
                    return host
            except Exception:
                pass
    if debug_row:
        for key in ['final_url', 'resolved_url', 'url']:
            u = debug_row.get(key)
            if u:
                try:
                    host = urlparse(str(u)).netloc
                    if host:
                        return host
                except Exception:
                    pass
    return 'unknown'


def build_report(days=1):
    days = max(1, int(days))
    target_days = _target_days(days)
    days_set = set(target_days)

    by_day = _collect_processed(days_set)
    debug_idx = _collect_debug_index(days_set)

    lines = []
    all_totals = Counter()

    for day in target_days:
        rows = by_day.get(day, [])
        status_counter = Counter(str(r.get('status', 'unknown')) for r in rows)
        total = len(rows)

        ok = status_counter.get('ok', 0)
        skip_short = status_counter.get('skip_short', 0)
        skip_mismatch = status_counter.get('skip_mismatch', 0)
        error = status_counter.get('error', 0)
        duplicate = status_counter.get('duplicate', 0) + status_counter.get('skip_duplicate', 0)

        non_inflow = total - ok
        non_inflow_pct = _safe_pct(non_inflow, total)
        mismatch_pct = _safe_pct(skip_mismatch, total)

        fail_domains = Counter()
        fail_types = Counter()
        for r in rows:
            st = str(r.get('status', ''))
            if st == 'ok':
                continue
            dbg = debug_idx.get((day, r.get('url')))
            fail_domains[_domain_from_row(r, dbg)] += 1
            if st == 'error':
                fail_types[_error_type(r)] += 1
            else:
                fail_types[st] += 1

        top_domains = ', '.join(f'{k}({v})' for k, v in fail_domains.most_common(3)) or '-'
        top_types = ', '.join(f'{k}({v})' for k, v in fail_types.most_common(3)) or '-'

        lines.append(f'[뉴스봇2 추출 품질 리포트] {day.isoformat()}')
        lines.append(
            f'targets={total}, ok={ok}, skip_short={skip_short}, '
            f'skip_mismatch={skip_mismatch}, error={error}, duplicate={duplicate}'
        )
        lines.append(f'articles 비유입={non_inflow_pct:.1f}% (총처리-ok = {non_inflow})')
        lines.append(f'skip_mismatch={mismatch_pct:.1f}%')
        lines.append(f'top_fail_domains: {top_domains}')
        lines.append(f'top_fail_types: {top_types}')
        lines.append('')

        all_totals.update({
            'targets': total,
            'ok': ok,
            'skip_short': skip_short,
            'skip_mismatch': skip_mismatch,
            'error': error,
            'duplicate': duplicate,
        })

    if days > 1:
        total = all_totals['targets']
        non_inflow = total - all_totals['ok']
        lines.append(f'[요약 {days}일]')
        lines.append(
            f"targets={all_totals['targets']}, ok={all_totals['ok']}, "
            f"skip_short={all_totals['skip_short']}, skip_mismatch={all_totals['skip_mismatch']}, "
            f"error={all_totals['error']}, duplicate={all_totals['duplicate']}"
        )
        lines.append(f'articles 비유입={_safe_pct(non_inflow, total):.1f}% (총처리-ok = {non_inflow})')
        lines.append(f"skip_mismatch={_safe_pct(all_totals['skip_mismatch'], total):.1f}%")

    return '\n'.join(lines).strip() + '\n'


def write_report(text: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    kst_now = datetime.now(KST)
    stamp = kst_now.strftime('%Y-%m-%d')
    ts = kst_now.strftime('%Y%m%d_%H%M%S')

    dated = REPORT_DIR / f'extract_quality_{stamp}.txt'
    stamped = REPORT_DIR / f'extract_quality_{ts}.txt'
    latest = REPORT_DIR / 'extract_quality_latest.txt'

    dated.write_text(text, encoding='utf-8')
    stamped.write_text(text, encoding='utf-8')
    latest.write_text(text, encoding='utf-8')
    return [dated, stamped, latest]


def send_telegram(text: str):
    token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()
    if not token or not chat_id:
        print('[warn] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; skip telegram send')
        return False

    url = f'https://api.telegram.org/bot{token}/sendMessage'
    resp = requests.post(url, json={'chat_id': chat_id, 'text': text}, timeout=15)
    if not resp.ok:
        print(f'[warn] telegram send failed: {resp.status_code} {resp.text[:200]}')
        return False
    return True


def main():
    days = int(os.getenv('EXTRACT_REPORT_DAYS', '1'))
    to_tg = os.getenv('EXTRACT_REPORT_TO_TELEGRAM', '0').strip() in {'1', 'true', 'True', 'yes', 'YES'}

    report = build_report(days=days)
    paths = write_report(report)

    print(report)
    print('[report_saved]', ', '.join(str(p) for p in paths))

    if to_tg:
        sent = send_telegram(report)
        print('[telegram_sent]', sent)


if __name__ == '__main__':
    main()
