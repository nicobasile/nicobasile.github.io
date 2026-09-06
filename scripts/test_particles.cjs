/* Deterministic simulation/lifecycle regressions. No browser dependencies. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../assets/particles.js'), 'utf8');

function engine(mode = 'web', reduced = false, readingRect = null) {
  let now = 0, seed = 123, nextId = 0, allocations = 0, measurements = 0;
  const callbacks = new Map();
  function events(target = {}) {
    const handlers = new Map();
    target.addEventListener = (type, fn) => {
      if (!handlers.has(type)) handlers.set(type, []);
      handlers.get(type).push(fn);
    };
    target.emit = (type, event = {}) => (handlers.get(type) || []).forEach(fn => fn(event));
    return target;
  }
  const math = Object.create(Math);
  math.random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296);
  const ctx = new Proxy({}, { get: (o, key) => o[key] || (key === 'createLinearGradient'
    ? () => ({ addColorStop() {} }) : () => {}), set: (o, k, v) => (o[k] = v, true) });
  const canvas = { getContext: () => ctx, style: {} };
  for (const key of ['width', 'height']) {
    let value = 0;
    Object.defineProperty(canvas, key, { get: () => value, set: v => { value = v; allocations++; } });
  }
  const preference = events({ matches: reduced });
  const card = events({ contains: () => false, getBoundingClientRect: () => {
    measurements++; return { left: 400, right: 800, top: 250, bottom: 600 };
  } });
  const document = events({ hidden: false, readyState: 'complete',
    getElementById: id => id === 'particle-canvas' ? canvas : null,
    querySelector: selector => readingRect && selector.startsWith('article.') ? { getBoundingClientRect: () => readingRect } : null, querySelectorAll: selector => selector === '.post-card' ? [card] : [] });
  const window = events({ innerWidth: 1440, innerHeight: 900, devicePixelRatio: 2, matchMedia: () => preference });
  const sandbox = { Math: math, performance: { now: () => now }, window, document,
    localStorage: { getItem: () => mode, setItem() {} },
    requestAnimationFrame: fn => { callbacks.set(++nextId, fn); return nextId; },
    cancelAnimationFrame: id => callbacks.delete(id) };
  vm.createContext(sandbox);
  // Test-only seam: production has no debug globals or instrumentation overhead.
  vm.runInContext(source.replace('  function init() {', `
    globalThis.inspect = () => ({ particles, dragon, accumulator, articleFlight, step: STEP_MS });
    globalThis.testHead = updateDragonHead;
    globalThis.testRoute = (a, b) => articleRoute(a, b, articleRail());
    globalThis.testCrosses = (a, b) => crossesArticle(a, b, articleBounds(0));
    globalThis.testRender = render;
    globalThis.setMode = setAmbientMode;
    function init() {`), sandbox);
  return { window, document, preference, card, ctx, inspect: sandbox.inspect, render: sandbox.testRender, headStep: sandbox.testHead, route: sandbox.testRoute, crosses: sandbox.testCrosses, setMode: sandbox.setMode,
    pending: () => callbacks.size, allocations: () => allocations, measurements: () => measurements,
    frame(time) { now = time; const jobs = [...callbacks.values()]; callbacks.clear(); jobs.forEach(fn => fn(now)); },
    move(x = 1000, y = 450) { window.emit('pointermove', { clientX: x, clientY: y }); } };
}
function run(hz, mode, behavior, seconds = 10) {
  const e = engine(mode); e.frame(0);
  if (behavior === 'dragon') e.move();
  if (behavior === 'card') e.card.emit('mouseenter');
  for (let i = 1; i <= hz * seconds; i++) e.frame(i * 1000 / hz);
  return e;
}
function snapshot(e) {
  return JSON.parse(JSON.stringify({ particles: [...e.inspect().particles].sort((a,b) => a.index-b.index)
    .map(p => [p.x,p.y,p.vx,p.vy,p.baseVx,p.baseVy,p.dragonWeight]),
    head: e.inspect().dragon.head, time: e.inspect().dragon.time,
    sleep: e.inspect().dragon.sleepWeight, idle: e.inspect().dragon.idleWeight }));
}
function near(a,b, label) {
  if (typeof a === 'number') assert.ok(Math.abs(a-b) < 1e-7, `${label}: ${a} != ${b}`);
  else for (const key of Object.keys(a)) near(a[key], b[key], `${label}.${key}`);
}
for (const mode of ['web', 'flow']) for (const behavior of ['ambient', 'dragon', 'card']) {
  const reference = run(60, mode, behavior, 30);
  for (const hz of [30,120,144]) near(snapshot(run(hz, mode, behavior, 30)), snapshot(reference), `${mode}/${behavior}/${hz}`);
  console.log(`PASS equal 30-second simulation at 30/60/120/144 Hz: ${mode}/${behavior}`);
}
const variable = engine(); variable.frame(0);
let t = 0;
for (const hz of [120,30,60,30,120,60]) {
  for (let i = 1; i <= hz; i++) variable.frame(t + i * 1000 / hz);
  t += 1000;
}
near(snapshot(variable), snapshot(run(60, 'web', 'ambient', 6)), 'changing rate');
const beforeDraw = snapshot(variable);
variable.render(0.25); variable.render(0.75);
near(snapshot(variable), beforeDraw, 'drawing never changes physics');
const p = variable.inspect().particles[0];
p.previousX = 10; p.x = 20; p.previousY = 40; p.y = 60;
variable.render(0.5); assert.equal(p.renderX,15); assert.equal(p.renderY,50);
p.x = -26; p.vx = -1; p.update();
assert.equal(p.previousX,p.x); variable.render(0.5); assert.equal(p.renderX,p.x);
const stall = engine(); stall.frame(0); stall.frame(10000);
assert.ok(Math.abs(stall.inspect().dragon.time - 6 * 0.016) < 1e-8);
stall.frame(10000 + 1000/60);
assert.ok(Math.abs(stall.inspect().dragon.time - 7 * 0.016) < 1e-8);
const life = run(60, 'web', 'dragon', 5);
assert.ok(life.inspect().dragon.rig.length > 0);
for (let i=0;i<10;i++) { life.document.emit('visibilitychange'); life.window.emit('pageshow'); }
assert.equal(life.pending(),1);
life.document.hidden = true; life.document.emit('visibilitychange'); assert.equal(life.pending(),0);
const hiddenState = snapshot(life); life.frame(60000); near(snapshot(life),hiddenState,'hidden');
life.document.hidden = false; life.document.emit('visibilitychange'); life.frame(60001);
near(snapshot(life),hiddenState,'resume without catch-up'); assert.equal(life.pending(),1);
const allocated = life.allocations();
for(let i=0;i<20;i++) life.window.emit('resize');
life.frame(60010); assert.equal(life.allocations(),allocated);
life.window.innerWidth = 390; life.window.emit('resize'); life.frame(60011);
assert.equal(life.inspect().particles.length,60);
assert.equal(life.inspect().dragon.rig.length,0); assert.equal(life.inspect().dragon.members.length,0);
assert.equal(life.inspect().dragon.history.length,0); assert.equal(life.inspect().dragon.rigCooldown,0);
life.preference.matches = true; life.preference.emit('change'); assert.equal(life.pending(),0);
life.preference.matches = false; life.preference.emit('change'); assert.equal(life.pending(),1);
life.frame(70000); assert.equal(life.inspect().dragon.time,0);
const reduced = engine('web',true); assert.equal(reduced.pending(),0);
reduced.preference.matches=false; reduced.preference.emit('change'); reduced.frame(0);
assert.equal(reduced.inspect().particles.length,120);
life.frame(70000 + 1000/60);
life.window.innerWidth = 1440; life.window.emit('resize'); life.frame(70050);
assert.equal(life.inspect().particles.length,120);
for (let i=1;i<=300;i++) life.frame(70050 + i*1000/60);
assert.ok(life.inspect().dragon.rig.every(e => life.inspect().particles.includes(e.p)));
assert.ok(life.inspect().dragon.members.every(p => life.inspect().particles.includes(p)));
life.setMode('flow'); life.frame(75100); life.setMode('web'); life.frame(75150);
assert.ok(life.inspect().particles.every(p => Number.isFinite(p.x) && Number.isFinite(p.y)));
const hover = engine(); hover.frame(0); hover.card.emit('mouseenter');
const measured = hover.measurements();
for (let i=1;i<=60;i++) hover.frame(i*1000/60);
assert.equal(hover.measurements(),measured, 'stable card geometry should be cached');
hover.window.emit('scroll'); hover.frame(1050); assert.equal(hover.measurements(),measured+1);
hover.card.emit('mouseleave'); hover.frame(1100);
const sleep = run(60,'web','dragon',25); assert.equal(sleep.inspect().dragon.sleepWeight,1);
sleep.move(1010,450); sleep.frame(25000 + 1000/60); assert.ok(sleep.inspect().dragon.sleepWeight < 1);
console.log('PASS rate changes, render purity/interpolation/wrapping, bounded stalls, visibility, duplicate events, resize/rig reset, reduced motion, sleep/wake');

// Exercise steering without recruitment resetting the head to its seed particle.
const chase = engine(); chase.frame(0); chase.move(1100, 450);
const dragon = chase.inspect().dragon;
Object.assign(dragon.head, { x: 300, y: 450, vx: 0, vy: 0 });
for (let i = 1; i <= 100; i++) chase.headStep(i * chase.inspect().step);
assert.ok(dragon.head.x > 450, 'distant dragon makes direct progress toward cursor');
assert.equal(dragon.head.y, 450, 'no circular sideways drift during distant pursuit');
assert.equal(dragon.pursuing, true);
// Even after the sleep timeout, finish the approach before idling.
chase.headStep(30000);
assert.equal(dragon.sleepWeight, 0);
assert.equal(dragon.idleWeight, 0);
// Within the transition band, the orbit starts to return gradually.
Object.assign(dragon.head, { x: 900, y: 450, vx: 0, vy: 0 });
chase.headStep(1000);
assert.ok(dragon.head.vx > 0 && Math.abs(dragon.head.vy) > 0);
assert.equal(dragon.pursuing, true);
// At the original orbit radius the original tangential steering is unchanged.
Object.assign(dragon.head, { x: 1100 - dragon.orbitRadius, y: 450, vx: 0, vy: 0 });
chase.headStep(1000);
assert.equal(dragon.pursuing, false);
assert.equal(dragon.head.vx, 0);
assert.ok(Math.abs(Math.abs(dragon.head.vy) - dragon.orbitSpeed * 0.045) < 1e-10);
// A new distant target exits idle choreography and redirects the dragon.
dragon.idleWeight = 1;
chase.move(100, 100);
chase.headStep(1000);
assert.equal(dragon.pursuing, true);
assert.ok(dragon.idleWeight < 1);
chase.card.emit('mouseenter'); chase.headStep(1000);
assert.equal(dragon.pursuing, false, 'card hover takes priority over pursuit');
console.log('PASS direct pursuit, smooth orbit return, distant idle suppression, retargeting, and hover priority');


for (const side of [-1, 1]) {
  const rect = {left:350,right:1090,top:160,bottom:4000,width:740};
  const reading = engine('web', false, rect); reading.frame(0);
  reading.move(side < 0 ? 160 : 1280, 450);
  for (let i=1;i<=180;i++) reading.frame(i*1000/90);
  const d = reading.inspect().dragon;
  Object.assign(d.head, {x:side < 0 ? 180 : 1250,y:450,vx:0,vy:1});
  // Enter on the opposite half of the text: the dragon's side wins.
  reading.move(side < 0 ? 1000 : 420,450);
  let minY=Infinity,maxY=-Infinity;
  for (let i=1;i<=400;i++) {
    if(i===180) reading.move(side < 0 ? 420 : 1000,550);
    if(i===240) { rect.top=-1500; reading.window.emit('scroll'); }
    reading.frame(2000+i*1000/90);
    assert.equal(reading.inspect().articleFlight.side,side);
    assert.ok(side < 0 ? d.head.x < rect.left : d.head.x > rect.right);
    minY=Math.min(minY,d.head.y);maxY=Math.max(maxY,d.head.y);
    assert.equal(d.sleepWeight,0,'no article perch');
  }
  assert.ok(maxY-minY>40,'dragon follows a curve in the gutter');
  assert.ok(d.members.length>0,'dragon is still assembled before five seconds');
  for(let i=401;i<=520;i++) reading.frame(2000+i*1000/90);
  assert.equal(d.members.length,0);
  assert.equal(d.rig.length,0);
  assert.ok(reading.inspect().particles.every(p=>!p.isDragonMember),'spine and appendages all release');
  assert.ok(reading.inspect().particles.every(p=>!(p.x>rect.left&&p.x<rect.right&&p.y>rect.top&&p.y<rect.bottom)));
  reading.move(720,600);reading.frame(8000);
  assert.equal(d.members.length,0,'motion within the article must not reassemble the dragon');
  reading.move(side < 0 ? 150 : 1280,450);
  reading.frame(8100);
  assert.equal(reading.inspect().articleFlight,null);
  assert.ok(d.members.length>0,'leaving the article restores normal recruitment');
}
const soft = engine('web',false,{left:350,right:1090,top:160,bottom:740,width:740});
soft.frame(0);const dust=soft.inspect().particles[0];
Object.assign(dust,{x:336,y:450,vx:1,vy:0,baseVx:1,baseVy:0,radius:2});
soft.frame(soft.inspect().step);
assert.ok(dust.vx>0&&dust.vx<0.9,'soft dust boundary remains');
const phone=engine('web',false,{left:20,right:370,top:140,bottom:4000,width:350});
phone.window.innerWidth=390;phone.frame(0);phone.move(10,450);phone.frame(20);
phone.move(200,450);
for(let i=1;i<=600;i++) phone.frame(20+i*1000/90);
assert.equal(phone.inspect().dragon.members.length,0);
assert.equal(phone.inspect().dragon.rig.length,0);
assert.ok(phone.inspect().particles.every(p=>!(p.x>20&&p.x<370&&p.y>140&&p.y<4000)));
console.log('PASS left/right gutter curves, five-second release, no hover re-recruitment, exit recovery, scrolling and mobile boundaries');
