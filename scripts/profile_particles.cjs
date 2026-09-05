/* Usage: NODE_PATH=... node scripts/profile_particles.cjs BEFORE AFTER OUTPUT_DIR
 * Isolated headed Chromium, local builds, fixed viewport. Trace files open in DevTools.
 * Profiling is injected only into these pages; production scripts have no telemetry.
 */
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
if (process.argv.length !== 5) throw Error('Usage: node scripts/profile_particles.cjs BEFORE AFTER OUTPUT_DIR');
const [before, after, output] = process.argv.slice(2).map(p => path.resolve(p));
fs.mkdirSync(output, { recursive: true });
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
async function endTrace(cdp,file) {
  const done=new Promise(r=>cdp.once('Tracing.tracingComplete',r));
  await cdp.send('Tracing.end'); const {stream}=await done;
  const fd=fs.openSync(file,'w');
  for(;;){const r=await cdp.send('IO.read',{handle:stream}); fs.writeSync(fd,r.base64Encoded?Buffer.from(r.data,'base64'):r.data); if(r.eof)break;}
  fs.closeSync(fd); await cdp.send('IO.close',{handle:stream});
}
(async()=>{
  const servers=[await serve(before),await serve(after)];
  const browser=await chromium.launch({headless:false});
  const results=[];
  try {
    for (let version=0;version<2;version++) {
      if (process.env.VERSION && process.env.VERSION !== (version ? "after" : "before")) continue;
      const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:2});
      await context.addInitScript(()=>{
        window.__profile={on:false,frames:[],sim:0,draw:0,last:null,visibility:[]};
        document.addEventListener('visibilitychange',()=>window.__profile.visibility.push({hidden:document.hidden,time:performance.now()}));
        const raf=window.requestAnimationFrame;
        window.requestAnimationFrame=function(callback){
          if(callback.name!=='loop')return raf.call(window,callback);
          return raf.call(window,function(t){
            const p=window.__profile,begin=performance.now(); p.sim=p.draw=0;
            callback(t);
            if(p.on){p.frames.push({gap:p.last===null?null:t-p.last,total:performance.now()-begin,sim:p.sim,draw:p.draw});p.last=t;}
          });
        };
      });
      await context.route('**/assets/particles.js',async route=>{
        const response=await route.fetch();let source=await response.text();
        const fixed=source.includes('function simulate(now)');
        let injection=`\n function measured(fn,kind){ return function(...args){ const begin=performance.now(); try{return fn.apply(this,args);}finally{window.__profile[kind]+=performance.now()-begin;} }; }\n`;
        if(fixed) injection+='simulate=measured(simulate,"sim"); render=measured(render,"draw");\n';
        else {
          injection+='updateDragonHead=measured(updateDragonHead,"sim"); updateDragonMembers=measured(updateDragonMembers,"sim"); updateRig=measured(updateRig,"sim");\n';
          injection+='drawConnections=measured(drawConnections,"draw");drawDragon=measured(drawDragon,"draw");drawParticles=measured(drawParticles,"draw");\n';
          source=source.replace('    for (let i = 0; i < particles.length; i++) {\n      particles[i].update();\n    }', '    const updateStart=performance.now();\n    for (let i = 0; i < particles.length; i++) { particles[i].update(); }\n    window.__profile.sim+=performance.now()-updateStart;');
        }
        source=source.replace('  function init() {',injection+'  function init() {');
        await route.fulfill({response,body:source});
      });
      const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
      for(const [name,urlPath] of [['home','/'],['article','/robotic-foundation-model-training/']]) {
        if (process.env.PAGE && process.env.PAGE !== name) continue;
        await page.goto(servers[version].url+urlPath);await page.waitForTimeout(2000);
        const cdp=await context.newCDPSession(page);
        const label=`${version?'after':'before'}-${name}`;
        const geometry=await page.evaluate(()=>{
          let reads=0;
          const original=Element.prototype.getBoundingClientRect;
          Element.prototype.getBoundingClientRect=function(){if(this.tagName==='VIDEO')reads++;return original.call(this);};
          try { window.SiteMedia.refresh(); } finally {Element.prototype.getBoundingClientRect=original;}
          return {videoGeometryReads:reads,registeredVideos:document.querySelectorAll('video[data-media]').length};
        });
        console.log(JSON.stringify({label,...geometry}));
        await cdp.send('Tracing.start',{categories:'devtools.timeline,blink.user_timing,cc,gpu,media',transferMode:'ReturnAsStream'});
        async function sample(phase,duration,action) {
          await page.evaluate(phase=>{performance.mark(phase+'-start');Object.assign(window.__profile,{on:true,frames:[],last:null});},phase);
          if(action)await action();else await page.waitForTimeout(duration);
          const data=await page.evaluate(phase=>{performance.mark(phase+'-end');window.__profile.on=false;return {frames:window.__profile.frames,playing:[...document.querySelectorAll('video')].filter(v=>!v.paused).length,hidden:document.hidden};},phase);
          const item={label,phase,playing:data.playing,hidden:data.hidden,gap:stats(data.frames.map(f=>f.gap).filter(v=>v!==null)),callback:stats(data.frames.map(f=>f.total)),simulation:stats(data.frames.map(f=>f.sim)),drawing:stats(data.frames.map(f=>f.draw))};
          results.push(item); console.log(JSON.stringify(item));
        }
        await sample('moving',0,async()=>{for(let i=0;i<80;i++){await page.mouse.move(1150+Math.sin(i/10)*100,400+Math.cos(i/10)*120);await page.waitForTimeout(50);}});
        await sample('stationary',30000);
        if(name==='home') {
          await page.locator('.post-card').first().hover();await sample('card-hover',4000);
        }
        await sample('scrolling',0,async()=>{for(let i=0;i<30;i++){await page.mouse.wheel(0,i<15?140:-140);await page.waitForTimeout(100);}});
        await page.evaluate(()=>window.scrollTo(0,0));await page.mouse.move(1200,450);await page.waitForTimeout(1000);
        const style=await page.addStyleTag({content:'* {backdrop-filter:none!important;-webkit-backdrop-filter:none!important;}'});
        await sample('blur-off',4000);await style.evaluate(el=>el.remove());
        await page.evaluate(()=>{window.__play=HTMLMediaElement.prototype.play;HTMLMediaElement.prototype.play=function(){return Promise.resolve();};document.querySelectorAll('video').forEach(v=>v.pause());});
        await sample('media-paused',4000);
        await page.evaluate(async()=>{
          HTMLMediaElement.prototype.play=window.__play;
          // Restore only visible players; direct play clears the test's manual pause state.
          await Promise.all([...document.querySelectorAll('video')].filter(v=>{
            const r=v.getBoundingClientRect();return r.width>0&&r.height>0&&r.bottom>0&&r.top<innerHeight;
          }).map(v=>v.play().catch(()=>{})));
        });
        await sample('media-restored',8000);
        // Automation windows can keep both tabs visible. Attempt a real tab transition,
        // then explicitly label a simulated visibility event if that did not hide the page.
        const other=await context.newPage();await other.goto('about:blank');await other.bringToFront();await page.waitForTimeout(1000);
        const realHidden=await page.evaluate(()=>document.hidden);
        if(!realHidden) await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'));});
        await sample('background',2000);
        await other.close();await page.bringToFront();
        if(!realHidden)await page.evaluate(()=>{delete document.hidden;document.dispatchEvent(new Event('visibilitychange'));});
        await sample('resume',3000);
        await page.screenshot({path:path.join(output,label+'.png')});
        await endTrace(cdp,path.join(output,label+'.trace.json'));await cdp.detach();
        console.log(JSON.stringify({label,realBackgroundTransition:realHidden,errors}));
        if(errors.length)throw Error(errors.join('\n'));
      }
      await context.close();
    }
    fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify({browser:browser.version(),viewport:'1440x900 @2x',headed:true,results},null,2));
  } finally {await browser.close();servers.forEach(s=>s.server.close());}
})().catch(e=>{console.error(e);process.exitCode=1;});
