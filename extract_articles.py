#!/usr/bin/env python3
import os
import re
import requests, trafilatura
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from googlenewsdecoder import gnewsdecoder
from common import DATA, now_iso, read_jsonl, append_jsonl

IN = DATA / 'news_links.jsonl'
OUT = DATA / 'articles.jsonl'
PROCESSED = DATA / 'processed_news_links.jsonl'
DEBUG = DATA / 'extract_debug.jsonl'
MIN_TEXT_LEN = int(os.getenv('ARTICLE_MIN_TEXT_LEN', '120'))
TITLE_MATCH_MIN = int(os.getenv('ARTICLE_TITLE_MATCH_MIN', '2'))
TITLE_OVERLAP_MIN = float(os.getenv('ARTICLE_TITLE_OVERLAP_MIN', '0.12'))
TITLE_PAGE_SIM_MIN = float(os.getenv('ARTICLE_TITLE_PAGE_SIM_MIN', '0.32'))
MAX_REPEAT_RATIO = float(os.getenv('ARTICLE_REPEAT_RATIO_MAX', '0.28'))
TITLE_TOKEN_STOP = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'and', 'on', 'at', 'is', 'are', 'was', 'were',
                    'this', 'that', 'it', 'as', 'by', 'with', 'from', 'news', 'official', 'officials'}


def fallback_extract(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    txt = ' '.join(soup.get_text(' ', strip=True).split())
    return txt[:12000]



def _norm_text(s):
    return re.sub(r'\s+', ' ', (s or '').strip().lower())


def _tokens(s):
    t = re.sub(r"[^a-zA-Z0-9가-힣]", " ", s.lower())
    return [x for x in t.split() if len(x) > 1 and x not in TITLE_TOKEN_STOP]


def _overlap_ratio(a, b):
    at = set(_tokens(a))
    bt = set(_tokens(b))
    if not at or not bt:
        return 0.0
    return len(at & bt) / float(len(at))


def _contains_title(text, title):
    if not title:
        return True
    t = title.lower()
    txt = _norm_text(text).lower()
    toks = [x for x in _tokens(t) if len(x) > 1]
    if len(toks) <= TITLE_MATCH_MIN:
        return any(x in txt for x in toks)
    matched = sum(1 for x in toks if x in txt)
    return matched >= max(1, len(toks) // 2)


def _sim(a, b):
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


def _too_repetitive(text):
    words = [w for w in re.findall(r'[가-힣a-zA-Z0-9]+', text.lower()) if w]
    if not words:
        return False
    uniq = set(words)
    return (len(uniq) / float(len(words))) < MAX_REPEAT_RATIO


def _page_title_from_html(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        return ''
    return ''


def _looks_like_error_page(text, page_title=''):
    merged = _norm_text(f"{text or ''} {page_title or ''}")
    bad_markers = [
        'access denied', 'forbidden', 'not found', '404', '403', 'just a moment',
        'captcha', 'bot check', 'automated requests', 'verify you are human',
        '요청하신 페이지를 찾을 수 없습니다', '접근이 차단', '접근 거부', '로봇 인증', '로봇 체크',
        '오류가 발생', '서버 오류', '잠시 후 다시 시도',
    ]
    return any(m in merged for m in bad_markers)


def _title_consistency_gate(source_title, article_text, page_title):
    st = source_title or ''
    txt = article_text or ''
    pt = page_title or ''

    title_match = _contains_title(txt, st)
    overlap = _overlap_ratio(txt, st)
    # source_title vs page_title 유사도 (본문-페이지타이틀이 아니라, 원천 메타 타이틀 일치성)
    page_sim = _sim(st, pt) if pt else 0.0
    repetitive = _too_repetitive(txt)
    errorish = _looks_like_error_page(txt, pt)

    # 최소 한 축(title_match/overlap/page_sim)은 살아 있어야 한다.
    passed = (title_match or overlap >= TITLE_OVERLAP_MIN or page_sim >= TITLE_PAGE_SIM_MIN) and not errorish

    # 반복률이 높고 제목축이 모두 약하면 강제 실패
    if repetitive and (not title_match) and overlap < TITLE_OVERLAP_MIN and page_sim < TITLE_PAGE_SIM_MIN:
        passed = False

    return passed, {
        'title_match': title_match,
        'title_overlap': round(overlap, 3),
        'title_page_sim': round(page_sim, 3),
        'too_repetitive': repetitive,
        'error_like': errorish,
    }


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
            page_title = _page_title_from_html(html)

            src_title = r.get('title', '')
            gate_ok, gate = _title_consistency_gate(src_title, text or '', page_title)

            append_jsonl(DEBUG, {
                'ts': now_iso(),
                'url': url,
                'resolved_url': resolved_url,
                'final_url': final_url,
                'status_code': getattr(resp, 'status_code', None),
                'text_len': text_len,
                'source_title': src_title,
                'page_title': page_title,
                **gate,
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

            if not gate_ok:
                append_jsonl(PROCESSED, {
                    'ts': now_iso(),
                    'url': url,
                    'resolved_url': resolved_url,
                    'final_url': final_url,
                    'status': 'skip_mismatch',
                    'text_len': text_len,
                    'source_title': src_title,
                    'page_title': page_title,
                    **gate,
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
