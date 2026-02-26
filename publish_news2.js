const { chromium } = require('playwright');
const fs = require('fs');
require('dotenv').config({ path: '/home/ubuntu/threads-bot-news2/.env' });

const OWN_HANDLE = (process.env.THREADS_OWN_HANDLE || '').replace('@', '').trim();
if (!OWN_HANDLE) throw new Error('THREADS_OWN_HANDLE is required (.env)');
const PROFILE = `https://www.threads.com/@${OWN_HANDLE}`;
const LOG = '/home/ubuntu/threads-bot/threads_publish_full.log';
const POST_TEXT = process.argv.slice(2).join(' ') || `자동 테스트 ${new Date().toISOString()}`;
const MAX_ATTEMPTS = 2; // 최초 1회 + 실패 시 재시도 1회
const CDP_URL = process.env.THREADS_CDP_URL || 'http://127.0.0.1:9222/';

fs.writeFileSync(LOG, '');
const log = (m) => fs.appendFileSync(LOG, `[${new Date().toISOString()}] ${m}\n`);

const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
const firstLine = norm(POST_TEXT.split('\n')[0]);

async function topPostInfo(page) {
  return page.evaluate(() => {
    const a = Array.from(document.querySelectorAll('a')).find((x) => /\/post\//.test(x.getAttribute('href') || ''));
    if (!a) return { href: '', text: '' };
    const h = a.getAttribute('href') || '';
    const href = h.startsWith('http') ? h : `https://www.threads.com${h}`;
    const article = a.closest('article');
    const text = (article?.innerText || a.innerText || '').replace(/\s+/g, ' ').trim();
    return { href, text };
  });
}

async function publishOnce(page, attempt) {
  await page.goto(PROFILE, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2000);

  const before = await topPostInfo(page);
  log(`ATTEMPT_${attempt}_TOP_BEFORE=${before.href || 'NONE'}`);

  // 중복 방지: 최상단 게시물 텍스트에 동일 본문(첫 줄)이 이미 있으면 중단
  if (firstLine && norm(before.text).includes(firstLine)) {
    log(`DUPLICATE_BLOCKED=${firstLine}`);
    return { ok: true, skipped: true, postUrl: before.href || '' };
  }

  // 작성 버튼: 언어/레이아웃 차이를 고려해 다중 셀렉터 시도
  const composeCandidates = [
    '[aria-label*="compose" i]',
    '[aria-label*="new thread" i]',
    '[aria-label*="new post" i]',
    '[aria-label*="create" i]',
    '[aria-label*="글쓰기" i]',
    '[aria-label*="새 스레드" i]',
    '[aria-label*="게시" i]',
    'a[href="/intent/post"]',
    'button:has-text("게시")',
    'button:has-text("Post")',
  ];

  let opened = false;
  for (const sel of composeCandidates) {
    const loc = page.locator(sel).first();
    if (await loc.count()) {
      try {
        await loc.click({ force: true, timeout: 3000 });
        opened = true;
        break;
      } catch {}
    }
  }
  if (!opened) {
    // 최후 fallback: /intent/post 직접 이동
    await page.goto('https://www.threads.com/intent/post', { waitUntil: 'domcontentloaded' });
  }
  await page.waitForTimeout(900);

  const dialog = page.locator('[role="dialog"]').last();
  const box = dialog.locator('[role="textbox"][contenteditable="true"]').first();
  await box.waitFor({ state: 'visible', timeout: 15000 });
  await box.click({ force: true });
  await box.fill('');
  await box.type(POST_TEXT, { delay: 18 });
  log(`ATTEMPT_${attempt}_TEXT_TYPED`);

  const postBtn = dialog.getByRole('button', { name: /^(Post|게시)$/i }).last();
  await postBtn.waitFor({ state: 'visible', timeout: 10000 });
  await postBtn.click({ force: true });
  log(`ATTEMPT_${attempt}_POST_CLICKED`);

  await page.waitForTimeout(2500);

  let found = '';
  for (let i = 0; i < 12; i++) {
    await page.goto(PROFILE, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1800);
    const nowTop = await topPostInfo(page);
    log(`ATTEMPT_${attempt}_CHECK_${i + 1}=${nowTop.href || 'NONE'}`);

    if (nowTop.href && nowTop.href !== before.href) {
      found = nowTop.href;
      break;
    }
    await page.waitForTimeout(1200);
  }

  if (!found) return { ok: false, skipped: false, postUrl: '' };
  return { ok: true, skipped: false, postUrl: found };
}

(async () => {
  const browser = await chromium.connectOverCDP(CDP_URL);
  const ctx = browser.contexts()[0];
  const page = ctx.pages()[0] || await ctx.newPage();

  let lastErr = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const result = await publishOnce(page, attempt);
      if (result.ok) {
        if (result.skipped) {
          console.log('POST_SKIPPED_DUPLICATE', result.postUrl || 'NO_URL');
          log(`POST_SKIPPED_DUPLICATE=${result.postUrl || 'NO_URL'}`);
        } else {
          console.log('POST_URL', result.postUrl);
          log(`POST_URL=${result.postUrl}`);
        }
        process.exit(0);
      }

      log(`ATTEMPT_${attempt}_FAILED_POST_NOT_FOUND`);
      if (attempt < MAX_ATTEMPTS) {
        log('RETRYING_ONCE');
        await page.waitForTimeout(4000);
      }
    } catch (e) {
      lastErr = e;
      log(`ATTEMPT_${attempt}_ERROR=${e?.message || e}`);
      if (attempt < MAX_ATTEMPTS) {
        log('RETRYING_ONCE_AFTER_ERROR');
        await page.waitForTimeout(4000);
      }
    }
  }

  log(`POST_URL=NOT_FOUND`);
  console.log('POST_URL', 'NOT_FOUND');
  if (lastErr) console.error(lastErr);
  process.exit(1);
})();
