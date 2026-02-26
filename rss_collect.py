#!/usr/bin/env python3
import os, re, feedparser
from common import DATA, now_iso, read_jsonl, append_jsonl, keywords_from_env

OUT = DATA / 'news_links.jsonl'


def main():
    feeds = [x.strip() for x in os.getenv('RSS_FEEDS', '').split(',') if x.strip()]
    include = [k.lower() for k in keywords_from_env('NEWS_INCLUDE_KEYWORDS')]
    exclude = [k.lower() for k in keywords_from_env('NEWS_EXCLUDE_KEYWORDS')]

    seen = set((r.get('url') or '').strip() for r in read_jsonl(OUT))
    added = 0

    for feed_url in feeds:
        d = feedparser.parse(feed_url)
        for e in d.entries:
            url = (getattr(e, 'link', '') or '').strip()
            title = (getattr(e, 'title', '') or '').strip()
            summary = re.sub('<[^<]+?>', ' ', getattr(e, 'summary', '') or '')
            text = f"{title} {summary}".lower()
            if not url or url in seen:
                continue
            if include and not any(k in text for k in include):
                continue
            if exclude and any(k in text for k in exclude):
                continue

            append_jsonl(OUT, {
                'ts': now_iso(),
                'feed': feed_url,
                'title': title,
                'url': url,
            })
            seen.add(url)
            added += 1

    print('RSS_COLLECT_OK', 'added', added, 'total', len(seen))


if __name__ == '__main__':
    main()
