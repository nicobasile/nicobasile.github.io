/* NODE_PATH=/path/to/node_modules node scripts/test_media.cjs /path/to/built-site [baseline-site] */
const { chromium, firefox, webkit } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const site = path.resolve(process.argv[2] || '_site');
const baseline = process.argv[3] && path.resolve(process.argv[3]);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.mp4': 'video/mp4', '.webp': 'image/webp', '.jpg': 'image/jpeg', '.png': 'image/png' };
function serve(root) {
  const server = http.createServer((req, res) => {
    let file = path.join(root, decodeURIComponent(req.url.split('?')[0]));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
    if (!fs.existsSync(file)) { res.writeHead(404).end(); return; }
    const size = fs.statSync(file).size;
    const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range || '');
    const start = range ? Number(range[1]) : 0;
    const end = range && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    const headers = { 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1 };
    if (range) headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
    res.writeHead(range ? 206 : 200, headers);
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(file, { start, end }).pipe(res);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` })));
}
const waitPlaying = (page, selector) => page.waitForFunction(s => {
  const v = document.querySelector(s); return v && !v.paused && v.currentTime > 0;
}, selector, { timeout: 15000 });
const waitPaused = (page, selector) => page.waitForFunction(s => [...document.querySelectorAll(s)].every(v => v.paused), selector);
async function verifyRecovery(browser, url, name) {
  let context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: name !== 'firefox', hasTouch: true });
  let page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('https://**/*', route => route.abort());
  const first = '[data-pair-pane="0"] video';
  let fail = true;
  await page.route('**/videos/side-1.mp4', async route => {
    if (fail) { fail = false; return route.fulfill({ status: 503, body: 'Temporary unavailable' }); }
    await new Promise(resolve => setTimeout(resolve, 800));
    await route.continue();
  });
  await page.goto(url + '/robotic-foundation-model-training/');
  await page.locator(first).first().scrollIntoViewIfNeeded();
  await waitPlaying(page, first);
  assert.equal(await page.locator(first).first().getAttribute('poster'), '/videos/posters/side-1.webp');
  await page.unroute('**/videos/side-1.mp4');
  // Use a fresh context: Firefox retains decoded media across navigations.
  await context.close();
  context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: name !== 'firefox', hasTouch: true });
  page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  await page.route('https://**/*', route => route.abort());
  // Real failed responses exhaust the automatic reload budget, then explicit Retry recovers.
  await page.addInitScript(() => {
    const load = HTMLMediaElement.prototype.load;
    window.mediaLoads = {};
    HTMLMediaElement.prototype.load = function () {
      const key = this.dataset.src;
      window.mediaLoads[key] = (window.mediaLoads[key] || 0) + 1;
      return load.call(this);
    };
  });
  await page.route('**/videos/side-1.mp4', route => route.fulfill({ status: 503, body: 'Unavailable' }));
  await page.goto(url + '/robotic-foundation-model-training/');
  await page.locator(first).first().scrollIntoViewIfNeeded();
  await page.waitForFunction(() => window.mediaLoads['/videos/side-1.mp4'] === 3, null, { timeout: 10000 }).catch(async e => { console.log(await page.evaluate(() => ({loads:window.mediaLoads,videos:[...document.querySelectorAll('video')].map(v=>({error:v.error && {code:v.error.code,message:v.error.message},paused:v.paused,button:v.parentElement.querySelector('.media-action')?.textContent}))}))); throw e; });
  await page.waitForTimeout(2500);
  assert.equal(await page.evaluate(() => window.mediaLoads['/videos/side-1.mp4']), 3, 'retry budget exceeded');
  assert.equal(await page.locator('[data-pair-pane="0"] .media-action').first().textContent(), 'Retry');
  assert.equal(await page.locator(first).count(), 2, 'video replaced on error');
  await page.unroute('**/videos/side-1.mp4');
  await page.locator('[data-pair-pane="0"] .media-action').first().click();
  await waitPlaying(page, first);
  // Load while offline, restore connectivity, then recover automatically.
  await context.setOffline(true);
  await page.evaluate(() => {
    const v = document.querySelector('[data-pair-pane="0"] video');
    v.src = v.dataset.src + '?offline-test=1';
    v.load();
    v.play().catch(() => {});
  });
  await page.waitForTimeout(500);
  await context.setOffline(false);
  await waitPlaying(page, first);
  // Below-fold images select WebP derivatives but expand to their original source.
  const inline = 'video[data-src="/videos/augmented_synthetic_data.mp4"]';
  await page.locator(inline).scrollIntoViewIfNeeded();
  await waitPlaying(page, inline);
  const img = page.locator('.post-entry figure img').first();
  await img.scrollIntoViewIfNeeded();
  await page.waitForFunction(() => { const i = document.querySelector('.post-entry figure img'); return i.complete && i.naturalWidth > 0; });
  assert((await img.evaluate(i => i.currentSrc)).includes('/images/responsive/'));
  const bounds = await img.boundingBox();
  assert(bounds.width <= 390, 'mobile image overflows viewport');
  const ratio = await img.evaluate(i => Number(i.getAttribute('width')) / Number(i.getAttribute('height')));
  assert(Math.abs(bounds.width / bounds.height - ratio) < 0.03, 'image aspect ratio changed');
  const original = await img.getAttribute('data-original');
  await img.click();
  assert((await page.locator('.lightbox img').getAttribute('src')).endsWith(original));
  await page.screenshot({ path: '/private/tmp/nico-media-' + name + '-expanded.png', fullPage: false });
  await page.locator('.lightbox-close').click();
  await page.screenshot({ path: '/private/tmp/nico-media-' + name + '-mobile.png', fullPage: false });
  assert.equal(errors.length, 0, errors.join('\n'));
  await context.close();
}
async function main() {
  const live = await serve(site);
  const before = baseline && await serve(baseline);
  const summary = [];
  try {
    for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
      if (process.env.BROWSERS && !process.env.BROWSERS.split(",").includes(name)) continue;
      console.log("Testing", name);
      const browser = await engine.launch();
      try {
        const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.route('https://**/*', route => route.abort());
        const requested = new Set();
        page.on('request', req => { if (req.url().endsWith('.mp4')) requested.add(path.basename(req.url())); });
        const article = '/robotic-foundation-model-training/';
        await page.goto(live.url + article);
        const first = '[data-pair-pane="0"] video';
        await page.locator(first).first().scrollIntoViewIfNeeded();
        await waitPlaying(page, first);
        assert(!requested.has('side-2.mp4') && !requested.has('top-2.mp4'), 'hidden comparisons fetched');
        assert(!requested.has('augmented_synthetic_data.mp4'), 'distant inline video fetched');
        const initial = [...requested];
        await page.screenshot({ path: '/private/tmp/nico-media-' + name + '-desktop.png' });
        for (const tab of [1, 0, 1, 0]) {
          await page.locator(`[data-pair-tab="${tab}"]`).click();
          await waitPlaying(page, `[data-pair-pane="${tab}"] video`);
          await waitPaused(page, `[data-pair-pane="${1-tab}"] video`);
        }
        await page.evaluate(() => scrollTo(0, document.body.scrollHeight));
        await waitPaused(page, first);
        await page.locator(first).first().scrollIntoViewIfNeeded();
        await waitPlaying(page, first);
        // A direct pause models the pause event from native media controls.
        await page.locator(first).first().evaluate(v => v.pause());
        await page.waitForTimeout(100);
        await page.evaluate(() => scrollTo(0, document.body.scrollHeight));
        await page.locator(first).first().scrollIntoViewIfNeeded();
        await page.waitForTimeout(200);
        assert(await page.locator(first).first().evaluate(v => v.paused), 'manual pause lost');
        await page.locator('[data-pair-pane="0"] .media-action').first().click();
        await waitPlaying(page, first);
        await page.locator('[data-pair-pane="0"] .media-zoom').first().click();
        await waitPlaying(page, '.lightbox video');
        await waitPaused(page, '.post-hero video');
        await page.locator('.lightbox-close').click();
        assert.equal(await page.locator('.lightbox video').count(), 0);
        await waitPlaying(page, first);
        // Deterministic visibility lifecycle; headless tabs do not consistently background.
        await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, value: true }); document.dispatchEvent(new Event('visibilitychange')); });
        await waitPaused(page, 'video');
        await page.evaluate(() => { delete document.hidden; document.dispatchEvent(new Event('visibilitychange')); });
        await waitPlaying(page, first);
        await page.goto(live.url + '/long-horizon-embodied-agents/');
        for (const clip of ['base', 'cactus', 'gold', 'pig']) {
          const selector = `.post-hero video[data-src="/videos/${clip}.mp4"]`;
          await page.locator(selector).scrollIntoViewIfNeeded();
          await waitPlaying(page, selector);
        }
        await page.goto(live.url);
        const cardVideo = '.post-card video';
        await page.locator(cardVideo).first().scrollIntoViewIfNeeded();
        await waitPlaying(page, cardVideo);
        const emptyFilter = 'foundation-models';
        await page.locator(`[data-filter="${emptyFilter}"]`).click();
        await waitPaused(page, cardVideo);
        await page.locator('[data-filter="all"]').click();
        await page.locator(cardVideo).first().scrollIntoViewIfNeeded();
        await waitPlaying(page, cardVideo);
        requested.clear();
        await page.goto(live.url + '/#foundation-models');
        await page.waitForTimeout(150);
        assert.equal(requested.size, 0, 'initial filtered cards fetched video');
        // Reduced motion: no MP4 request until explicit activation, no card navigation.
        await page.emulateMedia({ reducedMotion: 'reduce' });
        requested.clear();
        await page.goto(live.url);
        await page.locator(cardVideo).first().scrollIntoViewIfNeeded();
        await page.waitForTimeout(200);
        assert.equal(requested.size, 0);
        await page.locator('.post-card .media-action').first().click();
        await waitPlaying(page, cardVideo);
        assert.equal(page.url(), live.url + '/');
        await page.emulateMedia({ reducedMotion: 'no-preference' });
        // Autoplay policy rejection must retain player/poster and offer Play.
        await page.addInitScript(() => {
          const original = HTMLMediaElement.prototype.play;
          HTMLMediaElement.prototype.play = function () {
            if (!window.allowTestPlay) return Promise.reject(new DOMException('Blocked for test', 'NotAllowedError'));
            return original.call(this);
          };
        });
        await page.goto(live.url);
        await page.locator(cardVideo).first().scrollIntoViewIfNeeded();
        await page.locator('.post-card .media-action').first().waitFor({ state: 'visible' });
        assert.equal(await page.locator('.post-card .media-action').first().textContent(), 'Play');
        await page.evaluate(() => window.allowTestPlay = true);
        await page.locator('.post-card .media-action').first().click();
        await waitPlaying(page, cardVideo);
        assert.equal(errors.length, 0, errors.join('\n'));
        const baseRequests = new Set();
        if (before) {
          const old = await context.newPage();
          await old.route('https://**/*', route => route.abort());
          old.on('request', req => { if (req.url().endsWith('.mp4')) baseRequests.add(path.basename(req.url())); });
          await old.goto(before.url + article);
          await old.waitForTimeout(1000);
          await old.close();
        }
        summary.push({ browser: name, initialVideoRequests: initial, baselineVideoRequests: [...baseRequests], lifecycle: 'passed' });
        await context.close();
        await verifyRecovery(browser, live.url, name);
      } finally { await browser.close(); }
    }
    console.log(JSON.stringify(summary, null, 2));
  } finally { live.server.close(); if (before) before.server.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
