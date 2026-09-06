/* Headed browser regression: multiple low-FPS videos must not throttle the dragon.
 * Usage: NODE_PATH=... node scripts/test_particle_media.cjs BUILD_DIR
 * Optional BROWSER_CHANNEL=chrome runs the installed Google Chrome.
 */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const mime = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.mp4':'video/mp4', '.webp':'image/webp', '.png':'image/png', '.jpg':'image/jpeg', '.svg':'image/svg+xml' };
async function serve(root) {
  const server = http.createServer((req,res) => {
    let file = path.join(root,decodeURIComponent(req.url.split('?')[0]));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file=path.join(file,'index.html');
    if (!fs.existsSync(file)) { res.writeHead(404).end(); return; }
    const size=fs.statSync(file).size, range=/bytes=(\d+)-(\d*)/.exec(req.headers.range||'');
    const start=range?+range[1]:0, end=range&&range[2]?Math.min(+range[2],size-1):size-1;
    const headers={'Content-Type':mime[path.extname(file)]||'application/octet-stream','Content-Length':end-start+1,'Accept-Ranges':'bytes'};
    if(range) headers['Content-Range']=`bytes ${start}-${end}/${size}`;
    res.writeHead(range?206:200,headers); fs.createReadStream(file,{start,end}).pipe(res);
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  return {server,url:`http://127.0.0.1:${server.address().port}`};
}
function stats(values) {
  const v=values.filter(Number.isFinite).sort((a,b)=>a-b);
  return {count:v.length, mean:v.reduce((a,b)=>a+b,0)/(v.length||1),p50:v[Math.floor(v.length*.5)]||0,p95:v[Math.floor(v.length*.95)]||0,max:v.at(-1)||0};
}

(async () => {
  if (process.argv.length !== 3) throw Error('Usage: node scripts/test_particle_media.cjs BUILD_DIR');
  const { server, url } = await serve(path.resolve(process.argv[2]));
  let browser;
  try {
    browser = await chromium.launch({ headless: false, ...(process.env.BROWSER_CHANNEL ? { channel: process.env.BROWSER_CHANNEL } : {}) });
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
    await context.addInitScript(() => {
      window.frameTimes = [];
      const raf = window.requestAnimationFrame;
      window.requestAnimationFrame = callback => raf.call(window, t => {
        if (callback.name === 'loop') window.frameTimes.push(t);
        callback(t);
      });
    });
    for (const [route, comparison] of [['/', 0], ['/robotic-foundation-model-training/', 0], ['/robotic-foundation-model-training/', 1], ['/long-horizon-embodied-agents/', 0]]) {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(url + route);
      if (route !== '/') await page.locator('.post-hero').scrollIntoViewIfNeeded();
      if (comparison) await page.locator('[data-pair-tab="1"]').click();
      await page.mouse.move(1380, 450);
      await page.waitForFunction(() => [...document.querySelectorAll('video')].filter(v => !v.paused && v.readyState >= 2).length >= 2);
      async function sample(label, ms) {
        // Input temporarily boosts Chrome's cadence, masking the regression.
        await page.waitForTimeout(2500);
        await page.evaluate(() => { window.frameTimes = []; });
        await page.waitForTimeout(ms);
        const data = await page.evaluate(() => ({ times: window.frameTimes, hidden: document.hidden,
          playing: [...document.querySelectorAll('video')].filter(v => !v.paused).length }));
        assert.equal(data.hidden, false);
        const result = stats(data.times.slice(1).map((t, i) => t - data.times[i]));
        assert.ok(result.count > 30, 'particle loop must stay active');
        console.log(JSON.stringify({ route, comparison, label, playing: data.playing, ...result }));
        return result;
      }
      const playing = await sample('videos-playing', 8000);
      await page.evaluate(() => {
        window.playingVideos = [...document.querySelectorAll('video')].filter(v => !v.paused);
        window.playingVideos.forEach(v => v.pause());
      });
      const paused = await sample('videos-paused', 4000);
      // Compare against this machine's own cadence, not an assumed refresh rate.
      assert.ok(playing.mean < paused.mean * 1.35, `${route}: videos throttle mean cadence`);
      assert.ok(playing.p95 < Math.max(25, paused.p95 * 1.5), `${route}: videos introduce frame stalls`);
      assert.ok(playing.p95 < 35, `${route}: foreground animation must sustain smooth cadence`);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('PASS simultaneous video playback preserves particle frame cadence');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
