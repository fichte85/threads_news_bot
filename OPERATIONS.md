# threads-bot-news2 운영 가이드

## 1) 목적
해외 뉴스 수집 → 본문 추출 → 핫뉴스 선별 → 한국어 초안 생성 → 검수/선택 → 예약 발행.

---

## 2) 계정/런타임 분리 (핵심)
- 발행 계정: `news_fichte`
- 브라우저 프로필: `/home/ubuntu/.config/fichte_news`
- CDP 포트: `9223` (news2 전용)
- 서비스:
  - `news2-cdp.service` (크롬 9223)
  - `news2-watcher.service` (텔레그램 명령 처리)

### 발행 가드
`ensure_profile.py`가 발행 직전 아래를 검증:
1. 9223 CDP 크롬 프로세스 존재
2. `--user-data-dir=/home/ubuntu/.config/fichte_news`

불일치 시 발행 차단.

---

## 3) 디렉토리/파일
기준 경로: `/home/ubuntu/threads-bot-news2`

### 코드
- `rss_collect.py`
- `extract_articles.py`
- `hot_only.py`
- `generate_drafts.py`
- `watcher_bot.py`
- `schedule_publish.py`
- `ensure_profile.py`

### 데이터
- `data/news_links.jsonl`
- `data/processed_news_links.jsonl`
- `data/articles.jsonl`
- `data/extract_debug.jsonl`
- `data/drafts.jsonl`
- `data/publish_queue.json`
- `data/publish_state.json`

---

## 4) .env 핵심값
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `THREADS_PUBLISH_JS=/home/ubuntu/threads-bot/publish.js`
- `NEWS_THREADS_OWN_HANDLE=news_fichte`
- `NEWS_THREADS_CDP_URL=http://127.0.0.1:9223/`
- `GEN_PROVIDER=gemini`
- `GEMINI_MODEL=gemini-2.5-flash-lite`
- `HOT_TOP_N=10`
- `DRAFT_LIMIT=10`
- `DRAFT_MAX_CHARS=250`
- `MIX_CHAT_ENABLED` / `MIX_CHAT_EVERY`

---

## 5) 명령/버튼 운영 흐름

### 5-1) 수집+추출
- 명령: `/collect_extract`
- 내부 실행:
  - `python3 rss_collect.py`
  - `python3 extract_articles.py`

### 5-2) 생성
- 명령: `/generate_news`
- 내부 실행:
  - 기존 pending draft 자동 archive
  - `python3 hot_only.py` (Top N 선별)
  - `python3 generate_drafts.py` (초안 생성)

정책:
- 해시태그 금지/제거
- 스타일 참고 계정 기반(문장 복제 금지)

### 5-3) 검수
- `/review`: 번호 + 포맷 + 미리보기
- `/show <번호>`: 예약큐 번호 기준 상세(우선)
- `/show <draft_id>`: draft 원문 상세

### 5-4) 선택/예약
- `/pick 1,2,3`
- 예약큐 생성 규칙:
  - NEWS: 2시간 간격 정규 슬롯
  - CHAT: 뉴스 사이 비정규 삽입(ON일 때)

### 5-5) 예약 후 템플릿 재작성
- `/rework` → 큐 번호 목록 표시
- `/rework 2` → 템플릿 6개 번호 표시
- 번호 입력(1~6) 시 해당 큐 항목 text 덮어쓰기

템플릿:
1. PREP
2. PAS
3. LISTICLE
4. WHYHOW
5. BRIEF
6. INSIGHT

### 5-6) 발행
- 자동/수동 모두 프로필 가드 통과 필요
- 수동 테스트: `/publish_now <id>`
- 예약 발행: `schedule_publish.py` (queue head 기준)

---

## 6) 스케줄 로직
- NEWS는 2시간 간격 고정
- CHAT은 뉴스 사이 비정규 시간
- 기존 큐가 있을 때 새 pick 추가하면 기존 마지막 예약 뒤로 자동 이동

---

## 7) 안전 규칙
1. 민감정보 절대 노출 금지
2. 계정 분리 유지 (bot1과 news2 혼선 금지)
3. 발행 전 `/schedule` + `/show <큐번호>` 확인
4. API 429 시 `DRAFT_LIMIT`를 낮춰 소량 배치

---

## 8) 운영 루틴(권장)
1. `/collect_extract`
2. `/generate_news`
3. `/review`
4. `/pick ...` + 필요 시 `/rework ...`
5. `/schedule` 확인 후 대기

---

## 9) 잡담 운영
- 자동 잡담 삽입은 옵션
- 수동 개입 트리거 문구: **"뉴스 봇에 잡담 넣어줘"**
- 이 문구 수신 시, 큐 맥락에 맞춰 수동 잡담 작성/삽입
