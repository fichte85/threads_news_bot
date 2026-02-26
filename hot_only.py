#!/usr/bin/env python3
import os
import re
from math import exp
from collections import Counter, defaultdict
from urllib.parse import urlparse
from datetime import datetime, timezone
from common import DATA, read_jsonl, write_json, keywords_from_env

IN = DATA / 'articles.jsonl'
OUT = DATA / 'hot_candidates.json'

TOP_N = int(os.getenv('HOT_TOP_N', '10'))
HOT_MIN_SCORE = float(os.getenv('HOT_MIN_SCORE', '0'))

SOURCE_WEIGHT = {
    'reuters.com': 1.00,
    'apnews.com': 1.00,
    'bbc.com': 0.95,
    'bloomberg.com': 0.95,
    'wsj.com': 0.95,
    'nytimes.com': 0.95,
    'cnbc.com': 0.90,
    'cnn.com': 0.88,
    'techcrunch.com': 0.85,
    'theverge.com': 0.82,
}

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'for', 'of', 'in', 'on', 'with', 'at', 'by', 'from',
    'is', 'are', 'be', 'as', 'this', 'that', 'it', 'its', 'into', 'over', 'after', 'before',
    'ai', 'news', 'update', 'today', 'breaking', 'analysis',
    '및', '관련', '속보', '분석', '정리', '뉴스', '오늘',
}


def domain_of(url):
    try:
        d = urlparse(url or '').netloc.lower()
        return d.replace('www.', '')
    except Exception:
        return ''


def parse_dt(s):
    if not s:
        return None
    try:
        # fromisoformat에서 Z 보정
        s = str(s).replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def normalize_text(s):
    s = (s or '').lower()
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^0-9a-z가-힣 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def title_tokens(title):
    t = normalize_text(title)
    toks = [x for x in t.split() if len(x) >= 3 and x not in STOPWORDS]
    return toks


def topic_key(row):
    # 제목 토큰 상위 3개로 토픽 키 생성
    toks = title_tokens(row.get('source_title', ''))
    if not toks:
        # fallback: url path 마지막 식별자
        u = str(row.get('final_url') or row.get('resolved_url') or row.get('url') or '')
        m = re.search(r'/([^/?#]{6,})$', u)
        return m.group(1).lower() if m else u[:80].lower()
    return ' '.join(toks[:3])


def freshness_score(ts):
    dt = parse_dt(ts)
    if not dt:
        return 0.0
    now = datetime.now(timezone.utc)
    hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    # 24시간 기준 지수 감쇠
    return exp(-hours / 24.0)


def source_score(url):
    d = domain_of(url)
    if not d:
        return 0.2
    for dom, w in SOURCE_WEIGHT.items():
        if d.endswith(dom):
            return w
    return 0.35


def relevance_score(text, title, include_keywords):
    merged = normalize_text(f"{title} {text}")
    if not include_keywords:
        return 0.5
    hits = 0
    for kw in include_keywords:
        if normalize_text(kw) in merged:
            hits += 1
    return min(1.0, hits / max(1, min(len(include_keywords), 6)))


def build_buzz_models(rows):
    # 전체에서 자주 등장하는 신호어
    token_freq = Counter()
    topic_rows = defaultdict(list)
    for r in rows:
        toks = title_tokens(r.get('source_title', ''))
        token_freq.update(toks)
        topic_rows[topic_key(r)].append(r)
    return token_freq, topic_rows


def buzz_score(row, token_freq, topic_rows):
    # 토픽에 여러 출처가 몰리면 가점
    tk = topic_key(row)
    cluster = topic_rows.get(tk, [])
    domains = {domain_of(x.get('final_url') or x.get('resolved_url') or x.get('url')) for x in cluster}
    multi_source = min(1.0, max(0, len(domains) - 1) / 3.0)

    # 제목 토큰이 전체에서 자주 보이면 가점
    toks = title_tokens(row.get('source_title', ''))
    if toks:
        tf = sum(min(token_freq.get(t, 0), 6) for t in toks[:5]) / (len(toks[:5]) * 6)
    else:
        tf = 0.0

    return min(1.0, 0.6 * multi_source + 0.4 * tf)


def score_row(row, include_keywords, token_freq, topic_rows):
    text = row.get('text', '') or ''
    title = row.get('source_title', '') or ''
    f = freshness_score(row.get('ts'))
    s = source_score(row.get('final_url') or row.get('resolved_url') or row.get('url'))
    r = relevance_score(text, title, include_keywords)
    b = buzz_score(row, token_freq, topic_rows)

    # 가중치: freshness/buzz 중심
    total = 100.0 * (0.35 * f + 0.30 * b + 0.20 * s + 0.15 * r)
    return round(total, 2), {
        'freshness': round(f * 100, 1),
        'buzz': round(b * 100, 1),
        'source': round(s * 100, 1),
        'relevance': round(r * 100, 1),
    }


def main(top_n=TOP_N):
    rows = read_jsonl(IN)
    include_keywords = keywords_from_env('NEWS_INCLUDE_KEYWORDS')

    if not rows:
        write_json(OUT, {'generated_at': datetime.now().isoformat(), 'top_n': top_n, 'items': []})
        print('HOT_OK', 0, '->', OUT)
        return

    token_freq, topic_rows = build_buzz_models(rows)

    scored = []
    for r in rows:
        score, breakdown = score_row(r, include_keywords, token_freq, topic_rows)
        if score < HOT_MIN_SCORE:
            continue
        scored.append({
            'ts': r.get('ts'),
            'url': r.get('url'),
            'resolved_url': r.get('resolved_url'),
            'final_url': r.get('final_url'),
            'source_title': r.get('source_title', ''),
            'score': score,
            'topic_key': topic_key(r),
            'score_breakdown': breakdown,
        })

    scored.sort(key=lambda x: x['score'], reverse=True)

    # 다양성/중복 제약
    picked = []
    used_domains = set()
    used_topics = set()
    for s in scored:
        d = domain_of(s.get('final_url') or s.get('resolved_url') or s.get('url'))
        tk = s.get('topic_key', '')

        # 토픽 중복 방지 우선
        if tk and tk in used_topics:
            continue

        # 초반에는 도메인 다양성 강제
        if d in used_domains and len(picked) < max(3, top_n // 2):
            continue

        picked.append(s)
        if d:
            used_domains.add(d)
        if tk:
            used_topics.add(tk)
        if len(picked) >= top_n:
            break

    write_json(
        OUT,
        {
            'generated_at': datetime.now().isoformat(),
            'top_n': top_n,
            'min_score': HOT_MIN_SCORE,
            'weights': {'freshness': 0.35, 'buzz': 0.30, 'source': 0.20, 'relevance': 0.15},
            'items': picked,
        },
    )
    print('HOT_OK', len(picked), '->', OUT)


if __name__ == '__main__':
    main()
