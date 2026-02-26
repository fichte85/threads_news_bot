#!/usr/bin/env python3
import os
import requests, trafilatura
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder
from common import DATA, now_iso, read_jsonl, append_jsonl

IN = DATA / 'news_links.jsonl'
OUT = DATA / 'articles.jsonl'
PROCESSED = DATA / 'processed_news_links.jsonl'
DEBUG = DATA / 'extract_debug.jsonl'
MIN_TEXT_LEN = int(os.getenv('ARTICLE_MIN_TEXT_LEN', '120'))


def fallback_extract(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    txt = ' '.join(soup.get_text(' ', strip=True).split())
    return txt[:12000]


def resolve_google_news_url(url):
    u = str(url or '').strip()
    if 'news.google.com/rss/articles/' not in u:
        return u
    try:
        d = gnewsdecoder(u)
        if d.get('status') and d.get('decoded_url'):
            return d['decoded_url']
    except Exception:
        pass
    return u


def main(limit=60):
    rows = read_jsonl(IN)
    done = set(r.get('url') for r in read_jsonl(PROCESSED))
    targets = [r for r in rows if r.get('url') and r.get('url') not in done][:limit]

    ok = 0
    for r in targets:
        url = r['url']
        resolved_url = resolve_google_news_url(url)
        try:
            resp = requests.get(resolved_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            html = resp.text
            text = trafilatura.extract(html, include_comments=False, include_tables=False) or fallback_extract(html)
            text_len = len((text or '').strip())
            final_url = str(getattr(resp, 'url', resolved_url) or resolved_url)

            append_jsonl(DEBUG, {
                'ts': now_iso(),
                'url': url,
                'resolved_url': resolved_url,
                'final_url': final_url,
                'status_code': getattr(resp, 'status_code', None),
                'text_len': text_len,
            })

            if not text or text_len < MIN_TEXT_LEN:
                append_jsonl(PROCESSED, {
                    'ts': now_iso(),
                    'url': url,
                    'resolved_url': resolved_url,
                    'final_url': final_url,
                    'status': 'skip_short',
                    'text_len': text_len,
                    'min_len': MIN_TEXT_LEN,
                })
                continue

            append_jsonl(OUT, {
                'ts': now_iso(),
                'url': url,
                'resolved_url': resolved_url,
                'final_url': final_url,
                'source_title': r.get('title', ''),
                'text': text[:12000],
            })
            append_jsonl(PROCESSED, {
                'ts': now_iso(),
                'url': url,
                'resolved_url': resolved_url,
                'final_url': final_url,
                'status': 'ok',
                'text_len': text_len
            })
            ok += 1
        except Exception as e:
            append_jsonl(PROCESSED, {'ts': now_iso(), 'url': url, 'status': 'error', 'error': str(e)[:200]})

    print('EXTRACT_OK', ok, 'targets', len(targets), 'min_len', MIN_TEXT_LEN)


if __name__ == '__main__':
    main()
