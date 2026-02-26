# threads-bot-news2 (RSS → AI 요약 → 검수 → 예약발행)

2번째 Threads 계정용 뉴스 자동화 파이프라인.

## 흐름
1. RSS 수집 (`rss_collect.py`)
2. 기사 본문 추출 (`extract_articles.py`)
3. 핫뉴스 선별 (`hot_only.py`)
4. AI 초안 생성 (`generate_drafts.py`)
5. 텔레그램 검수/선택 (`watcher_bot.py`의 `/review`, `/pick`)
6. 예약 발행 (`schedule_publish.py`)

## 설치
```bash
cd /home/ubuntu/threads-bot-news2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 수동 실행
```bash
python3 rss_collect.py
python3 extract_articles.py
python3 generate_drafts.py
python3 watcher_bot.py
```

## 주요 명령(텔레그램)
- `/collect_news` : RSS 수집
- `/extract_news` : 본문 추출
- `/generate_news` : AI 초안 생성
- `/review` : 검수용 draft 목록 전송
- `/pick 12,15` : 선택 draft를 예약큐에 적재
- `/schedule` : 예약큐 확인
- `/status` : 파이프라인 상태
- `/publish_now 12` : 즉시 발행(테스트)

## 데이터 파일
- `data/news_links.jsonl`
- `data/processed_news_links.jsonl`
- `data/articles.jsonl`
- `data/processed_articles.jsonl`
- `data/drafts.jsonl`
- `data/publish_queue.json`
- `data/news2.log`

## 비고
- 발행은 `THREADS_PUBLISH_JS`(news2 전용 `publish_news2.js`)를 호출해 수행.
- 승인(pick)된 항목만 발행됨.
