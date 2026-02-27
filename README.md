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

## 추출 품질 리포트
- 스크립트: `report_extract_quality.py`
- 입력:
  - `data/processed_news_links.jsonl`
  - `data/extract_debug.jsonl` (보조)
- 출력:
  - `data/reports/extract_quality_YYYY-MM-DD.txt`
  - `data/reports/extract_quality_latest.txt`

### 환경변수
- `EXTRACT_REPORT_DAYS` (기본 `1`)
- `EXTRACT_REPORT_TO_TELEGRAM` (`0/1`, 기본 `0`)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 재사용

### 수동 실행
```bash
cd /home/ubuntu/threads-bot-news2
EXTRACT_REPORT_DAYS=1 EXTRACT_REPORT_TO_TELEGRAM=0 python3 report_extract_quality.py
```

### systemd timer 설치 예시 (매일 00:05 KST)
```bash
sudo cp deploy/systemd/news2-extract-quality-report.service /etc/systemd/system/
sudo cp deploy/systemd/news2-extract-quality-report.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now news2-extract-quality-report.timer
systemctl list-timers | grep news2-extract-quality-report
```

## 비고
- 발행은 `THREADS_PUBLISH_JS`(news2 전용 `publish_news2.js`)를 호출해 수행.
- 승인(pick)된 항목만 발행됨.
- `generate_drafts.py`는 `GEN_PROVIDER=writer`일 때 OpenClaw 에이전트 `writer(시드)`를 직접 호출해 초안을 생성함.
  - `.env`에서 `GEN_PROVIDER`를 `openai/anthropic/gemini`로 바꾸면 기존 방식으로 동작
