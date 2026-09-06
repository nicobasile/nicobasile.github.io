/**
 * Interactive Cosmic Gravity & Constellation Canvas Engine
 * Zero dependencies, battery-conscious, high-performance particle background.
 *
 * Behaviors:
 * 1. Background Cursor: Living Cosmic Dragon dynamically recruited from nearby ambient particles.
 *    Smoothly pathfinds via a pursuit-orbit spiral into a stable, fluid circular orbit around
 *    the cursor. Particles are dragged/flowed into the chain and then glide along the exact
 *    path the head traced, with wings that flap on a slow, independent cadence.
 * 2. Blog Post Card Hover: Pure rectangular gravity, solid box blocking & perimeter orbit.
 * 3. Ambient Dust & Idle: Fluid wake effects when the dragon glides past stars; graceful release
 *    and fast reset to neutral Brownian drift when idle.
 * 4. Ambient modes: Web (Brownian drift + constellation links) or Flow (shared curl field).
 * 5. Reading pages (article.post/page.detailed): dust and the dragon may travel
 *    around the article card, gliding in one gutter before dissolving while you read.
 */
(function () {
  'use strict';

  // Check user preference
  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  let enabled = !motionPreference.matches;

  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dragonCanvas = document.getElementById('dragon-canvas');
  const dragonCtx = dragonCanvas ? dragonCanvas.getContext('2d') : ctx;
  if (!ctx) return;
  const dustCtx = ctx;
  let animationFrameId = null;
  const STEP_MS = 1000 / 90;
  const MAX_STEPS = 6;
  let lastFrameAt = null;
  let accumulator = 0;
  let resizeDirty = true;
  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);

  // Graphite tones on the warm paper background.
  const PALETTE = ['#62625d', '#74746e', '#454540', '#555550', '#85857e'];
  const TWO_PI = Math.PI * 2;

  // Ambient field mode: 'web' = Brownian + constellation links (default),
  // 'flow' = shared curl so the dust streams in slow sheets.
  const MODE_KEY = 'particle_ambient_mode';
  let ambientMode = localStorage.getItem(MODE_KEY) === 'flow' ? 'flow' : 'web';

  function flowAt(x, y, t) {
    const s = 0.0024;
    const a = Math.sin(x * s + t * 0.14) + Math.cos(y * s * 1.25 - t * 0.11);
    const b = Math.cos(x * s * 0.85 - t * 0.09) - Math.sin(y * s + t * 0.12);
    return { x: b * 0.42, y: -a * 0.42 };
  }

  function syncModeSwitch() {
    const btn = document.getElementById('particle-mode');
    if (!btn) return;
    btn.setAttribute('aria-checked', ambientMode === 'flow' ? 'true' : 'false');
    btn.querySelectorAll('.particle-mode-opt').forEach((el) => {
      el.classList.toggle('is-active', el.getAttribute('data-mode') === ambientMode);
    });
  }

  function setAmbientMode(mode) {
    ambientMode = mode === 'flow' ? 'flow' : 'web';
    localStorage.setItem(MODE_KEY, ambientMode);
    syncModeSwitch();
  }

  // Particle configuration
  const config = {
    count: window.innerWidth < 640 ? 60 : 120,
    maxDistance: 95,
    damping: 0.965,
    ambientSpeed: 0.45
  };

  // State for active card, header, and reading column
  let activeCard = null;
  let activeCardRect = null;
  let isHeaderHovered = false;
  let cardRectDirty = false;

  // Reading pages: the article is a physical obstacle; the dragon follows its
  // current gutter briefly while the cursor is over the text.
  const readingEl = document.querySelector('article.post.detailed, article.page.detailed');
  const isReadingPage = !!readingEl;
  let readingDragonArmed = !isReadingPage;
  let articleRect = null;
  let articleRectDirty = true;
  let articleHover = false;
  let lastArticleSide = -1;
  let exitingArticle = new WeakSet();

  function refreshArticleHover() {
    const r = articleBounds(articleHover ? 14 : 6);
    const next = !!(r && mouse.active && insideRect(mouse, r));
    if (next && !articleHover) {
      exitingArticle = new WeakSet([dragon.head, ...particles.filter(p => p.isDragonMember)]);
    }
    articleHover = next;
    if (!next) articleFlight = null;
  }

  function dragonIgnoresArticle(p) {
    if (!articleHover) return true;
    if (!exitingArticle.has(p)) return false;
    const r = articleBounds((p.radius || 2) + 4);
    if (r) {
      if (insideRect(p, r)) return true;
      const edge = nearestEdge(p, r);
      if (p.vx * edge.nx + p.vy * edge.ny <= 0) return true;
    }
    exitingArticle.delete(p);
    return false;
  }


  function refreshArticleRect() {
    if (!readingEl) return;
    if (!articleRect || articleRectDirty) {
      articleRect = readingEl.getBoundingClientRect();
      articleRectDirty = false;
    }
  }

  function isOverArticle() {
    return articleHover;
  }

  function dragonCanHunt() {
    return readingDragonArmed && mouse.active && !activeCard && !isHeaderHovered &&
      (!articleFlight || !articleFlight.releasing);
  }

  const ARTICLE = { clearance: 48, cushion: 32, dragonCushion: 80, releaseDelay: 5000 };
  let articleFlight = null;

  function articleBounds(padding = 6) {
    if (!articleRect || articleRect.width === 0) return null;
    return { left: articleRect.left - padding, right: articleRect.right + padding,
      top: articleRect.top - padding, bottom: articleRect.bottom + padding };
  }

  // Sample clear viewport regions by area, rather than projecting random
  // starting points onto the wall. Include a gutter-scaled steering cushion.
  function placeInitialDust(p) {
    const r = articleBounds(p.radius + 4);
    if (!r) return;
    const cushion = space => Math.min(ARTICLE.cushion, Math.max(0, space) * 0.55);
    const left = Math.max(0, Math.min(width, r.left - cushion(r.left)));
    const right = Math.max(0, Math.min(width, r.right + cushion(width-r.right)));
    const top = Math.max(0, Math.min(height, r.top - cushion(r.top)));
    const bottom = Math.max(0, Math.min(height, r.bottom + cushion(height-r.bottom)));
    const regions = [[0,0,width,top], [0,bottom,width,height-bottom],
      [0,top,left,bottom-top], [right,top,width-right,bottom-top]];
    const area = regions.reduce((sum, r) => sum + r[2]*r[3], 0);
    if (area > 0) {
      let pick = Math.random()*area;
      for (const [x,y,w,h] of regions) {
        const size=w*h;
        if (size > 0 && pick < size) {
          p.x=x+Math.random()*w; p.y=y+Math.random()*h;
          break;
        }
        pick-=size;
      }
    } else {
      // No visible free space: keep dust offscreen instead of piling it on text.
      p.x = r.left - ARTICLE.cushion - 1;
      p.y = Math.random()*height;
    }
    p.previousX = p.renderX = p.x;
    p.previousY = p.renderY = p.y;
  }

  function insideRect(p, r) {
    return p.x > r.left && p.x < r.right && p.y > r.top && p.y < r.bottom;
  }

  function nearestEdge(p, r) {
    const choices = [
      { x: r.left, y: Math.max(r.top, Math.min(r.bottom, p.y)), nx: -1, ny: 0, edge: 'left' },
      { x: r.right, y: Math.max(r.top, Math.min(r.bottom, p.y)), nx: 1, ny: 0, edge: 'right' },
      { x: Math.max(r.left, Math.min(r.right, p.x)), y: r.top, nx: 0, ny: -1, edge: 'top' },
      { x: Math.max(r.left, Math.min(r.right, p.x)), y: r.bottom, nx: 0, ny: 1, edge: 'bottom' }
    ];
    return choices.reduce((a, b) => Math.hypot(p.x - a.x, p.y - a.y) <=
      Math.hypot(p.x - b.x, p.y - b.y) ? a : b);
  }

  function glanceDust(p, r, edge, distance, displaced) {
    const space = edge.edge === 'left' ? r.left : edge.edge === 'right' ? width - r.right :
      edge.edge === 'top' ? r.top : height - r.bottom;
    const cushion = Math.min(ARTICLE.cushion, Math.max(2, space * 0.55));
    if (!displaced && distance >= cushion) { p.dustGlance = null; return; }
    const tx = -edge.ny, ty = edge.nx;
    if (!p.dustGlance || p.dustGlance.edge !== edge.edge) {
      const inward = Math.min(p.vx * edge.nx + p.vy * edge.ny,
        p.baseVx * edge.nx + p.baseVy * edge.ny);
      if (!displaced && inward >= 0) return;
      const along = p.vx * tx + p.vy * ty;
      const baseAlong = p.baseVx * tx + p.baseVy * ty;
      const direction = Math.abs(along) > 0.001 ? Math.sign(along) :
        Math.abs(baseAlong) > 0.001 ? Math.sign(baseAlong) : (p.index % 2 ? 1 : -1);
      // Stable variation without consuming the simulation's random sequence.
      const angle = (25 + ((p.index * 7) % 16)) * Math.PI / 180;
      p.dustGlance = { edge: edge.edge, direction, angle };
    }
    const { direction, angle } = p.dustGlance;
    const dx = edge.nx * Math.sin(angle) + tx * direction * Math.cos(angle);
    const dy = edge.ny * Math.sin(angle) + ty * direction * Math.cos(angle);
    const heading = Math.atan2(dy, dx);
    // Rotate, don't blend vectors: blending can cancel velocity and park dust
    // against the wall. A displaced particle must depart immediately.
    for (const [x, y] of [['vx', 'vy'], ['baseVx', 'baseVy']]) {
      const speed = Math.hypot(p[x], p[y]);
      if (speed < 1e-12) continue;
      const current = Math.atan2(p[y], p[x]);
      const delta = Math.atan2(Math.sin(heading - current), Math.cos(heading - current));
      const limit = displaced ? Math.PI : Math.min(0.6, Math.max(0.12, speed * 1.6 / cushion));
      const turn = Math.max(-limit, Math.min(limit, delta));
      p[x] = Math.cos(current + turn) * speed;
      p[y] = Math.sin(current + turn) * speed;
    }
  }

  // Keep collision correction separate from glancing so geometry updates are
  // safe even on render frames that don't advance the simulation.
  function keepOutsideArticle(p, soft = false) {
    if ((p === dragon.head || p.isDragonMember) && dragonIgnoresArticle(p)) return;
    const r = articleBounds((p.radius || 2) + 4);
    if (!r) { if (soft) p.dustGlance = null; return; }
    if (p.isDragonMember) p.dustGlance = null;
    const margin = soft ? ARTICLE.cushion : 0;
    if (p.x < r.left - margin || p.x > r.right + margin ||
        p.y < r.top - margin || p.y > r.bottom + margin) {
      if (soft) p.dustGlance = null;
      return;
    }
    const edge = nearestEdge(p, r);
    const inside = insideRect(p, r);
    const distance = Math.hypot(p.x - edge.x, p.y - edge.y);
    if (inside) {
      p.x = edge.x; p.y = edge.y;
      p.previousX = p.x; p.previousY = p.y;
    }
    if (!p.isDragonMember && Number.isFinite(p.baseVx) && (soft || inside)) {
      // Runs after the Web/Flow drift update, preventing an inward field from
      // undoing the departure direction while the particle leaves the cushion.
      glanceDust(p, r, edge, distance, inside);
    } else if (inside) {
      // Preserve the dragon's existing emergency collision response exactly.
      const outward = p.vx * edge.nx + p.vy * edge.ny;
      if (outward < 0) { p.vx -= edge.nx * outward; p.vy -= edge.ny * outward; }
    }
  }

  function softenDragonWall(p) {
    if (dragonIgnoresArticle(p)) return;
    const r = articleBounds(8);
    if (!r || insideRect(p, r)) return;
    const edge = nearestEdge(p, r);
    const distance = Math.hypot(p.x - edge.x, p.y - edge.y);
    const gutter = edge.nx < 0 ? articleRect.left : edge.nx > 0 ? width - articleRect.right : height;
    const cushion = Math.min(ARTICLE.dragonCushion, Math.max(4, gutter * 0.4));
    if (distance >= cushion) return;
    const inward = p.vx * edge.nx + p.vy * edge.ny;
    if (inward >= 0) return;
    const t = 1 - distance / cushion;
    const turn = t * t * (3 - 2 * t) * 0.35;
    const tx = -edge.ny, ty = edge.nx;
    const along = p.vx * tx + p.vy * ty;
    const direction = Math.abs(along) > 0.01 ? Math.sign(along) : (dragon.head.vy >= 0 ? -1 : 1);
    // Redirect inward motion along the wall well before contact instead of
    // stopping every vertebra at the same hard boundary.
    p.vx += -inward * turn * (edge.nx + tx * direction);
    p.vy += -inward * turn * (edge.ny + ty * direction);
  }

  // Whether a segment crosses the rectangle interior (edge-following is OK).
  function crossesArticle(a, b, r) {
    let enter = 0, leave = 1;
    for (const [axis, low, high] of [['x', r.left + 0.01, r.right - 0.01],
      ['y', r.top + 0.01, r.bottom - 0.01]]) {
      const delta = b[axis] - a[axis];
      if (Math.abs(delta) < 1e-9) {
        if (a[axis] <= low || a[axis] >= high) return false;
      } else {
        const t1 = (low - a[axis]) / delta, t2 = (high - a[axis]) / delta;
        enter = Math.max(enter, Math.min(t1, t2));
        leave = Math.min(leave, Math.max(t1, t2));
      }
    }
    return enter < leave && leave > 0 && enter < 1;
  }

  function articleRail() {
    if (!articleRect) return null;
    // Fit the rail into narrow gutters rather than sending the head offscreen.
    const gap = (space) => Math.max(3, Math.min(ARTICLE.clearance, space / 2));
    return { left: articleRect.left - gap(articleRect.left),
      right: articleRect.right + gap(width - articleRect.right),
      top: articleRect.top - gap(articleRect.top),
      bottom: articleRect.bottom + gap(height - articleRect.bottom) };
  }

  function articleRoute(start, target, rail) {
    // Visibility graph: endpoints and the four article corners. Corners outside
    // the viewport are unavailable, so a long scrolled article keeps its reader
    // company in the current gutter until a visible route to the other opens.
    const obstacle = articleBounds(6);
    if (!crossesArticle(start, target, obstacle)) return { point: target, length: Math.hypot(target.x - start.x, target.y - start.y), direct: true };
    const nodes = [start, target, ...[
      { x: rail.left, y: rail.top }, { x: rail.right, y: rail.top },
      { x: rail.right, y: rail.bottom }, { x: rail.left, y: rail.bottom }
    ].filter(p => p.x >= 2 && p.x <= width - 2 && p.y >= 2 && p.y <= height - 2)];
    const distances = nodes.map(() => Infinity), previous = [], visited = [];
    distances[0] = 0;
    for (let k = 0; k < nodes.length; k++) {
      let u = -1;
      for (let i = 0; i < nodes.length; i++) if (!visited[i] && (u < 0 || distances[i] < distances[u])) u = i;
      if (u < 0 || !Number.isFinite(distances[u])) break;
      if (u === 1) {
        let next = 1;
        while (previous[next] !== 0) next = previous[next];
        return { point: nodes[next], length: distances[1], direct: false };
      }
      visited[u] = true;
      for (let v = 1; v < nodes.length; v++) {
        if (visited[v] || crossesArticle(nodes[u], nodes[v], obstacle)) continue;
        const distance = distances[u] + Math.hypot(nodes[v].x - nodes[u].x, nodes[v].y - nodes[u].y);
        if (distance < distances[v]) { distances[v] = distance; previous[v] = u; }
      }
    }
    return null;
  }

  function updateArticleFlight(now) {
    if (!readingDragonArmed || !isOverArticle() || !mouse.active || activeCard || isHeaderHovered) {
      articleFlight = null;
      return;
    }
    if (!articleFlight) {
      // Latch the dragon's side on entry, not the pointer's position in the text.
      const delta = Math.abs(dragon.head.x - articleRect.left) - Math.abs(dragon.head.x - articleRect.right);
      const side = Math.abs(delta) < 0.001 ? lastArticleSide : delta < 0 ? -1 : 1;
      lastArticleSide = side;
      const gutter = side < 0 ? articleRect.left : width - articleRect.right;
      const clearance = Math.min(64, gutter * 0.35);
      const lowX = side < 0 ? 12 : articleRect.right + clearance;
      const highX = side < 0 ? articleRect.left - clearance : width - 12;
      const clampX = x => Math.max(Math.min(lowX, highX), Math.min(Math.max(lowX, highX), x));
      const start = { x: dragon.head.x, y: dragon.head.y };
      const downRoom = height - 24 - start.y, upRoom = start.y - 24;
      let direction = dragon.head.vy >= 0 ? 1 : -1;
      if ((direction > 0 ? downRoom : upRoom) < 160) direction *= -1;
      const distance = Math.max(60, Math.min(520, direction > 0 ? downRoom : upRoom));
      const momentum = Math.hypot(dragon.head.vx, dragon.head.vy) || 1;
      // One open Bezier sweep, seeded from the current heading. No looping
      // phase or repeated orbit; the whole body has space to follow the turn.
      articleFlight = { side, startedAt: now, releasing: false, progress: 0,
        points: [start,
          { x: start.x + dragon.head.vx / momentum * 100,
            y: Math.max(24, Math.min(height - 24, start.y + dragon.head.vy / momentum * 100)) },
          { x: clampX(side < 0 ? lowX : highX), y: start.y + direction * distance * 0.65 },
          { x: clampX((lowX + highX) / 2), y: start.y + direction * distance }] };

    }
    // Pointer motion within the article does not restart the five-second timer.
    articleFlight.progress = Math.min(1, Math.max(0, (now - articleFlight.startedAt) / ARTICLE.releaseDelay));
    articleFlight.releasing = articleFlight.progress >= 1;
  }

  function glideBesideArticle() {
    const flight = articleFlight;
    const t = flight.progress, u = 1 - t;
    const [a, b, c, d] = flight.points;
    const x = u*u*u*a.x + 3*u*u*t*b.x + 3*u*t*t*c.x + t*t*t*d.x;
    const y = u*u*u*a.y + 3*u*u*t*b.y + 3*u*t*t*c.y + t*t*t*d.y;
    // Derivative in pixels per simulation step: feed-forward plus gentle
    // correction follows the open curve without racing a circling target.
    const rate = STEP_MS / ARTICLE.releaseDelay;
    const vx = (3*u*u*(b.x-a.x) + 6*u*t*(c.x-b.x) + 3*t*t*(d.x-c.x)) * rate;
    const vy = (3*u*u*(b.y-a.y) + 6*u*t*(c.y-b.y) + 3*t*t*(d.y-c.y)) * rate;
    dragon.pursuing = false;
    dragon.idleWeight = Math.max(0, dragon.idleWeight - 0.045);
    dragon.sleepWeight = Math.max(0, dragon.sleepWeight - 0.06);
    dragon.beatPhase += 0.024;
    steerHead(vx + (x - dragon.head.x) * 0.035,
      vy + (y - dragon.head.y) * 0.035, 0.07, dragon.pursuitSpeed);
  }

  // Mouse & Cosmic Dragon Engine
  const mouse = {
    x: 0,
    y: 0,
    active: false,
    hasMoved: false
  };

  function recordHeadHistory() {
    const last = dragon.history[0];
    if (!last || Math.hypot(dragon.head.x - last.x, dragon.head.y - last.y) > 1.6) {
      dragon.history.unshift({ x: dragon.head.x, y: dragon.head.y });
      const maxHistorySamples = (dragonLength + 2) * 12;
      if (dragon.history.length > maxHistorySamples) {
        dragon.history.length = maxHistorySamples;
      }
    }
  }

  function steerHead(desiredVx, desiredVy, steerForce, maxHeadSpeed) {
    const head = dragon.head;
    const desired = { x: head.x, y: head.y, vx: desiredVx, vy: desiredVy };
    if (!dragonIgnoresArticle(head)) softenDragonWall(desired);
    desiredVx = desired.vx; desiredVy = desired.vy;
    const oldSpeed = Math.hypot(head.vx, head.vy);
    const targetSpeed = Math.hypot(desiredVx, desiredVy);
    let vx = head.vx + (desiredVx - head.vx) * steerForce;
    let vy = head.vy + (desiredVy - head.vy) * steerForce;
    let speed = Math.hypot(vx, vy);
    if (oldSpeed > 0.01 && targetSpeed > 0.01) {
      // Blend speed independently: opposing direction vectors must not cancel
      // the dragon's forward motion halfway through a bank.
      speed = oldSpeed + (targetSpeed - oldSpeed) * steerForce;
      const heading = Math.atan2(head.vy, head.vx);
      let targetTurn = Math.atan2(head.vx * desiredVy - head.vy * desiredVx,
        head.vx * desiredVx + head.vy * desiredVy);
      // At an ambiguous 180-degree reversal, keep banking in the current
      // direction instead of letting tiny rounding changes flip the choice.
      if (Math.abs(targetTurn) > Math.PI - 0.08) {
        targetTurn = Math.abs(targetTurn) * Math.sign(dragon.turnVelocity || dragon.spinDir);
      }
      let wantedTurn = Math.atan2(head.vx * vy - head.vy * vx, head.vx * vx + head.vy * vy);
      if (Math.abs(targetTurn) > Math.PI / 2) {
        // Bank into a new mode rather than braking to zero and flipping over.
        wantedTurn = targetTurn * steerForce;
      }
      wantedTurn = Math.max(-dragon.maxTurnRate, Math.min(dragon.maxTurnRate, wantedTurn));
      dragon.turnVelocity += Math.max(-dragon.maxTurnAcceleration,
        Math.min(dragon.maxTurnAcceleration, wantedTurn - dragon.turnVelocity));
      const turn = dragon.turnVelocity;
      speed = Math.min(speed, maxHeadSpeed);
      vx = Math.cos(heading + turn) * speed;
      vy = Math.sin(heading + turn) * speed;
    } else {
      dragon.turnVelocity = 0;
      if (speed > maxHeadSpeed) { vx *= maxHeadSpeed / speed; vy *= maxHeadSpeed / speed; }
    }
    head.vx = vx; head.vy = vy;
    head.x += head.vx; head.y += head.vy;
    keepOutsideArticle(head);
    recordHeadHistory();
  }

  let dragonLength = window.innerWidth < 640 ? 14 : 20;

  const dragon = {
    head: { x: 0, y: 0, vx: 0, vy: 0 },
    history: [], // [{x, y}], stores exact path samples recorded by the head
    members: [], // Dynamically recruited Particle instances [P_head, ..., P_tail]
    orbitRadius: 78,
    orbitSpeed: 1.6, // Relaxed, tranquil orbit speed
    pursuitStartDistance: 300, // Chase directly when the cursor is this far away (CSS px)
    pursuitEndDistance: 140, // Fully return to orbit inside this distance
    pursuitSpeed: 2.4,
    pursuing: false,
    maxTurnRate: 0.055, // Radians per simulation step: limits sudden heading changes
    maxTurnAcceleration: 0.0035, // Ease into/out of a bank across mode changes
    turnVelocity: 0,
    spinDir: 1, // 1 = clockwise, -1 = counter-clockwise
    activeWeight: 0,
    time: 0,
    recruitCooldown: 0,
    // Idle choreography: large lemniscate (infinity) centered on the resting cursor
    idleWeight: 0,
    figTheta: 0,
    figA: 200, // half-width of the infinity
    figB: 110, // vertical lobe amplitude
    figSpeed: 1.9,
    figDir: 1, // travel direction along the curve, latched for the whole idle run
    // Dragon animation clocks
    beatPhase: 0, // slow wingbeat clock (drives the wings only, not flight speed)
    sleepWeight: 0, // 1 = curled up asleep
    rig: [], // appendage particles: [{ p, slot, settle }]
    rigCooldown: 0
  };

  const IDLE_DELAY_MS = 3800; // let it circle a while before the infinity kicks in
  const SLEEP_DELAY_MS = 20000; // after this long untouched, it curls up and dozes
  let lastMoveAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());

  // Gerono lemniscate (figure-8), centered at origin:
  //   x(t) = A * sin(t),  y(t) = B * sin(t) * cos(t)
  // Crosses its own center at t = 0 and t = PI, which is exactly the cursor.
  function figPoint(t, A, B) {
    const sinT = Math.sin(t);
    return { x: A * sinT, y: B * sinT * Math.cos(t) };
  }

  function figTangent(t, A, B) {
    // d/dt of the above: (A cos t, B cos 2t)
    const dx = A * Math.cos(t);
    const dy = B * Math.cos(2 * t);
    const mag = Math.hypot(dx, dy) || 0.001;
    return { x: dx / mag, y: dy / mag, mag };
  }

  // Find the parameter t whose curve point is nearest a local offset (ox, oy).
  // Used to "pathfind" into the shape from wherever the head currently is.
  function figClosestTheta(ox, oy, A, B) {
    let bestT = 0;
    let bestD = Infinity;
    const samples = 64;
    for (let i = 0; i < samples; i++) {
      const t = (i / samples) * Math.PI * 2;
      const p = figPoint(t, A, B);
      const d = (p.x - ox) * (p.x - ox) + (p.y - oy) * (p.y - oy);
      if (d < bestD) {
        bestD = d;
        bestT = t;
      }
    }
    return bestT;
  }

  const pointAttractor = {
    x: 0,
    y: 0,
    active: false
  };

  const DRAGON = {
    headColor: '#454540',
    midColor: '#74746e',
    tailColor: '#62625d',
    crestColor: '#555550',
    snoutLen: 20
  };

  // The dragon is a constellation: the spine is the recruited particle chain,
  // and these slots are extra particles pulled out to form wings and horns.
  // Every line is drawn between real particle positions, so nothing exists on
  // screen until the particle that defines it has actually flown into place.
  const WINGS = [
    { anchor: 3, trail: 9, span: 48 } // one clear pair, swept back along the body
  ];
  const HORN_BACK = 7;
  const HORN_OUT = 12;

  // Slot table: snout, horns, then the wing elbow/tip joints per side
  const RIG_SLOTS = [{ kind: 'snout' }];
  for (let side = -1; side <= 1; side += 2) {
    RIG_SLOTS.push({ kind: 'horn', side: side });
  }
  for (let w = 0; w < WINGS.length; w++) {
    for (let side = -1; side <= 1; side += 2) {
      RIG_SLOTS.push({ kind: 'wing', wing: w, side: side, part: 'mid' });
      RIG_SLOTS.push({ kind: 'wing', wing: w, side: side, part: 'tip' });
    }
  }

  class Particle {
    constructor(index = 0, total = 100) {
      this.index = index;
      this.total = total;
      this.isDragonMember = false;
      this.dustGlance = null;
      this.dragonWeight = 0;
      this.memberIndex = -1; // position in the spine, or -1 if not a vertebra
      this.reset(true);
    }

    reset(initial = false) {
      this.x = Math.random() * width;
      this.y = initial ? Math.random() * height : (Math.random() < 0.5 ? -10 : height + 10);

      const angle = Math.random() * Math.PI * 2;
      const speed = (0.2 + Math.random() * 0.35) * config.ambientSpeed;
      this.baseVx = Math.cos(angle) * speed;
      this.baseVy = Math.sin(angle) * speed;

      this.previousX = this.renderX = this.x;
      this.previousY = this.renderY = this.y;
      this.vx = this.baseVx;
      this.vy = this.baseVy;

      this.radius = Math.random() * 1.5 + 1.1;
      this.colorIndex = Math.floor(Math.random() * PALETTE.length);
      this.color = PALETTE[this.colorIndex];
      this.alpha = Math.random() * 0.45 + 0.4;
      this.orbitOffset = (Math.random() - 0.5) * 22; // Particle-specific orbital track
      this.orbitSpeedFactor = 0.82 + Math.random() * 0.36;

      this.isDragonMember = false;
      this.dustGlance = null;
      this.dragonWeight = 0;
      this.memberIndex = -1;
      if (initial) placeInitialDust(this);
    }

    update() {
      // 1. Hovering a Blog Post Card: Pure Rectangular Gravity, Solid Box Blocking & Perimeter Orbit
      if (activeCard && activeCardRect) {
        const rect = activeCardRect;
        const pad = Math.max(8, 18 + this.orbitOffset);
        const boxLeft = rect.left - pad;
        const boxRight = rect.right + pad;
        const boxTop = rect.top - pad;
        const boxBottom = rect.bottom + pad;

        const cxEdge = Math.max(boxLeft, Math.min(this.x, boxRight));
        const cyEdge = Math.max(boxTop, Math.min(this.y, boxBottom));

        const dx = this.x - cxEdge;
        const dy = this.y - cyEdge;
        const distToBox = Math.sqrt(dx * dx + dy * dy);

        const isInside = this.x >= boxLeft && this.x <= boxRight && this.y >= boxTop && this.y <= boxBottom;

        if (isInside) {
          // Immediately eject particle to the nearest outer edge of the rectangle
          const dLeft = this.x - boxLeft;
          const dRight = boxRight - this.x;
          const dTop = this.y - boxTop;
          const dBottom = boxBottom - this.y;
          const minD = Math.min(dLeft, dRight, dTop, dBottom);

          if (minD === dLeft) {
            this.x = boxLeft;
            this.vx = Math.min(this.vx, 0) - 0.8;
            this.vy -= 0.3;
          } else if (minD === dRight) {
            this.x = boxRight;
            this.vx = Math.max(this.vx, 0) + 0.8;
            this.vy += 0.3;
          } else if (minD === dTop) {
            this.y = boxTop;
            this.vy = Math.min(this.vy, 0) - 0.8;
            this.vx += 0.3;
          } else {
            this.y = boxBottom;
            this.vy = Math.max(this.vy, 0) + 0.8;
            this.vx -= 0.3;
          }
        } else {
          const nx = distToBox > 0.001 ? dx / distToBox : 0;
          const ny = distToBox > 0.001 ? dy / distToBox : -1;

          // Tangential direction: 90 deg clockwise around the rectangle
          const tx = -ny;
          const ty = nx;

          // Inward gravity toward nearest edge
          const slowGravity = 0.075;
          this.vx -= nx * slowGravity;
          this.vy -= ny * slowGravity;

          // Perimeter conveyer swirl
          const swirlFactor = Math.max(0.12, 1 - distToBox / 420);
          const swirlStrength = 0.38 * swirlFactor * this.orbitSpeedFactor;
          this.vx += tx * swirlStrength;
          this.vy += ty * swirlStrength;

          // Boundary cushion
          const cushion = 34;
          if (distToBox < cushion && distToBox > 0.001) {
            const pushFactor = (1 - distToBox / cushion) * 0.65;
            this.vx += nx * pushFactor;
            this.vy += ny * pushFactor;
          }
        }
      } else if (pointAttractor.active) {
        // 2. Hovering navigation links
        const dx = pointAttractor.x - this.x;
        const dy = pointAttractor.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const pull = 0.055;
        this.vx += (dx / dist) * pull;
        this.vy += (dy / dist) * pull;
      }

      // 3. Speed limiter & damping per mode
      if (activeCard) {
        // Blog post card hover - keep exact current good speed and fluid damping
        const maxSpeed = 3.4;
        const currentSpeed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
        if (currentSpeed > maxSpeed) {
          this.vx = (this.vx / currentSpeed) * maxSpeed;
          this.vy = (this.vy / currentSpeed) * maxSpeed;
        }
        const currentDamping = 0.978;
        this.vx = this.vx * currentDamping + this.baseVx * (1 - currentDamping);
        this.vy = this.vy * currentDamping + this.baseVy * (1 - currentDamping);
      } else if (this.isDragonMember) {
        // Dynamically recruited dragon member: fluid damping & speed cap
        // (headroom above the head's cruise speed so the tail never detaches)
        const maxSpeed = 4.2;
        const currentSpeed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
        if (currentSpeed > maxSpeed) {
          this.vx = (this.vx / currentSpeed) * maxSpeed;
          this.vy = (this.vy / currentSpeed) * maxSpeed;
        }
        const w = Math.min(1.0, Math.max(0.2, this.dragonWeight));
        const dragonDamping = 0.92;
        const ambientDamping = 0.88;
        const effDamping = ambientDamping * (1 - w) + dragonDamping * w;
        // Pure velocity decay for dragon members so ambient drift doesn't bias the trailing vertebrae
        this.vx = this.vx * effDamping + this.baseVx * 0.04 * (1 - w);
        this.vy = this.vy * effDamping + this.baseVy * 0.04 * (1 - w);
      } else if (ambientMode === 'flow') {
        const f = flowAt(this.x, this.y, dragon.time);
        const mix = 0.06;
        if (this.dustGlance) {
          // Keep Flow's speed response, but let the active departure own direction.
          // Blending opposing field vectors here would brake dust before steering.
          const baseSpeed = Math.hypot(this.baseVx, this.baseVy);
          const speed = Math.hypot(this.vx, this.vy);
          const nextBase = baseSpeed + (Math.hypot(f.x, f.y) * this.orbitSpeedFactor - baseSpeed) * mix;
          const nextSpeed = speed + (nextBase - speed) * 0.08;
          if (baseSpeed > 1e-12) {
            this.baseVx *= nextBase / baseSpeed;
            this.baseVy *= nextBase / baseSpeed;
          }
          if (speed > 1e-12) {
            this.vx *= nextSpeed / speed;
            this.vy *= nextSpeed / speed;
          }
        } else {
          this.baseVx += (f.x * this.orbitSpeedFactor - this.baseVx) * mix;
          this.baseVy += (f.y * this.orbitSpeedFactor - this.baseVy) * mix;
          this.vx += (this.baseVx - this.vx) * 0.08;
          this.vy += (this.baseVy - this.vy) * 0.08;
        }
      } else {
        // Neutral chaos state / Ambient stars: Rapidly reset velocity back to ambient Brownian drift
        const resetSpeed = 0.12;
        this.vx += (this.baseVx - this.vx) * resetSpeed;
        this.vy += (this.baseVy - this.vy) * resetSpeed;

        // Subtle organic wander so neutral particles drift naturally in all directions
        this.baseVx += (Math.random() - 0.5) * 0.03;
        this.baseVy += (Math.random() - 0.5) * 0.03;
        const speed = Math.sqrt(this.baseVx * this.baseVx + this.baseVy * this.baseVy);
        const targetSpeed = 0.42;
        if (speed > 0.001) {
          this.baseVx = (this.baseVx / speed) * targetSpeed;
          this.baseVy = (this.baseVy / speed) * targetSpeed;
        }
      }

      this.x += this.vx;
      this.y += this.vy;

      // 5. Viewport boundary wrapping
      // A wrap is a discontinuity: never interpolate across the viewport.
      if (this.x < -25 || this.x > width + 25) {
        this.x = this.x < -25 ? width + 25 : -25;
        this.previousX = this.x;
        this.previousY = this.y;
      }
      if (this.y < -25 || this.y > height + 25) {
        this.y = this.y < -25 ? height + 25 : -25;
        this.previousX = this.x;
        this.previousY = this.y;
      }
    }

    // Vertebrae swell slightly so they read as scale glints along the spine
    radiusNow() {
      if (this.isDragonMember && this.dragonWeight > 0.05) {
        // memberIndex is stamped once per frame; an indexOf here would make
        // drawing O(particles * spine length).
        const memberIdx = this.memberIndex;
        const ratio = memberIdx >= 0 ? memberIdx / Math.max(1, dragon.members.length) : 0.5;
        const targetRad = this.radius * (1.35 - 0.45 * ratio);
        return this.radius * (1 - this.dragonWeight) + targetRad * this.dragonWeight;
      }
      return this.radius;
    }
  }

  let particles = [];
  let renderOrder = [];

  // Dots are grouped by palette colour and quantised opacity, then each group
  // is filled as a single path. Otherwise every particle costs its own
  // fillStyle change, beginPath and fill.
  const ALPHA_STEPS = 4;
  const dotBatches = [];
  for (let c = 0; c < PALETTE.length * ALPHA_STEPS; c++) dotBatches.push([]);

  function drawParticles(foreground = false) {
    const ctx = foreground ? dragonCtx : dustCtx;
    for (let b = 0; b < dotBatches.length; b++) dotBatches[b].length = 0;

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (!!p.isDragonMember !== foreground) continue;
      const step = Math.min(ALPHA_STEPS - 1, (p.alpha * ALPHA_STEPS) | 0);
      const batch = dotBatches[p.colorIndex * ALPHA_STEPS + step];
      batch.push(p.renderX, p.renderY, p.radiusNow());
    }

    for (let b = 0; b < dotBatches.length; b++) {
      const dots = dotBatches[b];
      if (dots.length === 0) continue;
      ctx.fillStyle = PALETTE[(b / ALPHA_STEPS) | 0];
      ctx.globalAlpha = ((b % ALPHA_STEPS) + 0.5) / ALPHA_STEPS;
      ctx.beginPath();
      for (let k = 0; k < dots.length; k += 3) {
        // moveTo before each arc keeps the sub-paths from being joined
        ctx.moveTo(dots[k] + dots[k + 2], dots[k + 1]);
        ctx.arc(dots[k], dots[k + 1], dots[k + 2], 0, TWO_PI);
      }
      ctx.fill();
    }
  }

  function resetDragon() {
    dragon.members = [];
    dragon.rig = [];
    dragon.history = [];
    dragon.recruitCooldown = dragon.rigCooldown = 0;
    dragon.activeWeight = dragon.idleWeight = dragon.sleepWeight = 0;
    dragon.figTheta = dragon.beatPhase = dragon.time = 0;
    dragon.spinDir = dragon.figDir = 1;
    dragon.turnVelocity = 0;
    dragon.pursuing = false;
    articleFlight = null;
    dragon.head.x = width / 2;
    dragon.head.y = height / 2;
    dragon.head.vx = dragon.head.vy = 0;
  }

  function resize() {
    resizeDirty = false;
    const nextWidth = window.innerWidth;
    const nextHeight = window.innerHeight;
    const nextDpr = Math.min(window.devicePixelRatio || 1, 2);
    if (nextWidth === width && nextHeight === height && nextDpr === dpr) return;
    width = nextWidth;
    height = nextHeight;
    dpr = nextDpr;
    const pixelWidth = Math.floor(width * dpr);
    const pixelHeight = Math.floor(height * dpr);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (dragonCanvas) {
      if (dragonCanvas.width !== pixelWidth) dragonCanvas.width = pixelWidth;
      if (dragonCanvas.height !== pixelHeight) dragonCanvas.height = pixelHeight;
      dragonCanvas.style.width = width + 'px';
      dragonCanvas.style.height = height + 'px';
      dragonCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    refreshArticleRect();
    dragonLength = width < 640 ? 14 : 20;
    const targetCount = width < 640 ? 60 : 120;
    if (particles.length !== targetCount) {
      resetDragon();
      particles = Array.from({ length: targetCount }, (_, i) => new Particle(i, targetCount));
      renderOrder = particles.slice();
      accumulator = 0;
    }
    for (const p of particles) {
      p.previousX = p.renderX = p.x;
      p.previousY = p.renderY = p.y;
    }
  }

  // Physics uses its own sweep. Drawing must never change velocity or simulation order.
  function applyCardRepulsion() {
    if (!activeCard) return;
    particles.sort(sortByX);
    for (let i = 0; i < particles.length; i++) {
      const p1 = particles[i];
      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dx = p1.x - p2.x;
        if (-dx >= 24) break;
        const dy = p1.y - p2.y;
        const distSq = dx * dx + dy * dy;
        if (distSq >= 24 * 24 || distSq <= 0.000001) continue;
        const dist = Math.sqrt(distSq);
        const strength = (1 - dist / 24) * 0.38;
        const rx = dx / dist * strength;
        const ry = dy / dist * strength;
        p1.vx += rx;
        p1.vy += ry;
        p2.vx -= rx;
        p2.vy -= ry;
      }
    }
  }

  // Opacity buckets for batched constellation strokes, allocated once
  const LINK_BUCKETS = 5;
  const linkBuckets = [];
  for (let b = 0; b < LINK_BUCKETS; b++) linkBuckets.push([]);
  const sortByX = (a, b) => a.x - b.x;
  const sortByRenderX = (a, b) => a.renderX - b.renderX;

  function drawConnections() {
    // 1. Recruitment beams. Adjacent vertebrae are no longer stroked here - the
    //    dragon's ribbon body (drawDragon) renders the spine itself.
    if (dragon.members.length > 1) {
      const ctx = dragonCtx;
      ctx.save();
      for (let i = 1; i < dragon.members.length; i++) {
        const p1 = dragon.members[i - 1];
        const p2 = dragon.members[i];
        const d = Math.hypot(p2.renderX - p1.renderX, p2.renderY - p1.renderY);

        if (d >= 36 && dragon.activeWeight > 0.05) {
          // Being pulled from ambient space toward the dragon: subtle magnetic attraction beam
          const pullBeamAlpha = 0.16 * Math.max(0.1, 1 - d / 300) * dragon.activeWeight;
          if (pullBeamAlpha > 0.01) {
            ctx.beginPath();
            ctx.moveTo(p1.renderX, p1.renderY);
            ctx.lineTo(p2.renderX, p2.renderY);
            ctx.strokeStyle = '#74746e';
            ctx.lineWidth = 0.85;
            ctx.globalAlpha = pullBeamAlpha;
            ctx.stroke();
          }
        }
      }
      ctx.restore();
    }

    // 2. Pairwise proximity constellation lines (Web mode only).
    //    Flow mode skips the web so the stream reads as sheets, not a mesh.
    if (ambientMode !== 'web') return;

    const maxDist = config.maxDistance;
    const maxDistSq = maxDist * maxDist;
    const len = particles.length;

    renderOrder.sort(sortByRenderX);

    for (let b = 0; b < LINK_BUCKETS; b++) linkBuckets[b].length = 0;

    for (let i = 0; i < len; i++) {
      const p1 = renderOrder[i];
      for (let j = i + 1; j < len; j++) {
        const p2 = renderOrder[j];
        const dx = p1.renderX - p2.renderX;
        if (-dx > maxDist) break; // sorted by x: nothing further right can reach
        const dy = p1.renderY - p2.renderY;
        if (dy > maxDist || dy < -maxDist) continue;
        const distSq = dx * dx + dy * dy;

        if (distSq < maxDistSq) {
          const dist = Math.sqrt(distSq);

          // Links touching the dragon are dimmed hard, otherwise the ambient
          // web smears across the silhouette and hides its shape.
          if (ambientMode === 'web') {
            const onDragon = p1.isDragonMember || p2.isDragonMember;
            const alpha = (1 - dist / maxDist) * (onDragon ? 0.05 : 0.20);
            const bucket = linkBuckets[Math.min(LINK_BUCKETS - 1, (alpha / 0.20 * LINK_BUCKETS) | 0)];
            bucket.push(p1.renderX, p1.renderY, p2.renderX, p2.renderY);
          }
        }
      }
    }

    ctx.lineWidth = 0.75;
    ctx.strokeStyle = '#74746e';
    for (let b = 0; b < LINK_BUCKETS; b++) {
      const seg = linkBuckets[b];
      if (seg.length === 0) continue;
      ctx.globalAlpha = ((b + 0.5) / LINK_BUCKETS) * 0.20;
      ctx.beginPath();
      for (let k = 0; k < seg.length; k += 4) {
        ctx.moveTo(seg[k], seg[k + 1]);
        ctx.lineTo(seg[k + 2], seg[k + 3]);
      }
      ctx.stroke();
    }
  }

  // ------------------------------------------------------------------
  // Cosmic Dragon: a constellation, not a sprite.
  // The spine is the recruited particle chain; wings and horns are extra
  // recruited particles held at rig slots. Every stroke here connects real
  // particle positions, and its opacity follows how well those particles have
  // settled - so the dragon draws itself as the swarm arrives.
  // ------------------------------------------------------------------

  // Local frame (position + tangent + normal) at a spine member
  function memberFrame(i, rendering = false) {
    const x = rendering ? 'renderX' : 'x';
    const y = rendering ? 'renderY' : 'y';
    const mem = dragon.members;
    const idx = Math.max(0, Math.min(mem.length - 1, i));
    const a = mem[Math.max(0, idx - 1)];
    const b = mem[Math.min(mem.length - 1, idx + 1)];
    let tx = a[x] - b[x]; // points toward the head
    let ty = a[y] - b[y];
    const m = Math.hypot(tx, ty) || 1;
    tx /= m;
    ty /= m;
    return { x: mem[idx][x], y: mem[idx][y], tx: tx, ty: ty, nx: -ty, ny: tx, p: mem[idx] };
  }

  // Where a rig particle should sit right now, in the spine's local frame
  function rigSlotTarget(slot, particle = dragon.head) {
    const target = rawRigSlotTarget(slot);
    const boundary = articleHover && !dragonIgnoresArticle(particle) ? articleBounds(8) : null;
    if (target && boundary) {
      target.x = Math.max(5, Math.min(width - 5, target.x));
      target.y = Math.max(5, Math.min(height - 5, target.y));
    }
    return target && boundary && insideRect(target, boundary) ? nearestEdge(target, boundary) : target;
  }

  function rawRigSlotTarget(slot) {
    const mem = dragon.members;
    if (slot.kind === 'snout' || slot.kind === 'horn') {
      if (mem.length < 3) return null;
      const f = memberFrame(0);
      // Face along travel when moving, else along the neck
      let dx = dragon.head.vx;
      let dy = dragon.head.vy;
      const dm = Math.hypot(dx, dy);
      if (dm < 0.05) {
        dx = f.tx;
        dy = f.ty;
      } else {
        dx /= dm;
        dy /= dm;
      }
      if (slot.kind === 'snout') {
        return { x: f.x + dx * DRAGON.snoutLen, y: f.y + dy * DRAGON.snoutLen };
      }
      const droop = 1 - 0.4 * dragon.sleepWeight;
      return {
        x: f.x - dy * slot.side * HORN_OUT * droop - dx * HORN_BACK,
        y: f.y + dx * slot.side * HORN_OUT * droop - dy * HORN_BACK
      };
    }

    const cfg = WINGS[slot.wing];
    if (mem.length <= cfg.trail) return null;
    const f = memberFrame(cfg.anchor);

    // Flap: wings sweep out on the downbeat and fold in on the upbeat,
    // and stay folded against the flank while the dragon sleeps.
    const flap = 0.5 + 0.5 * Math.sin(dragon.beatPhase - slot.wing * 0.6);
    const ext = (0.34 + 0.66 * flap) * (1 - 0.88 * dragon.sleepWeight);

    if (slot.part === 'mid') {
      // Elbow: out to the side, slightly ahead of the shoulder
      return {
        x: f.x + f.nx * slot.side * cfg.span * 0.52 * ext + f.tx * cfg.span * 0.12,
        y: f.y + f.ny * slot.side * cfg.span * 0.52 * ext + f.ty * cfg.span * 0.12
      };
    }
    // Tip: further out and swept firmly BACK along the body
    return {
      x: f.x + f.nx * slot.side * cfg.span * 0.92 * ext - f.tx * cfg.span * 0.62,
      y: f.y + f.ny * slot.side * cfg.span * 0.92 * ext - f.ty * cfg.span * 0.62
    };
  }

  // Recruit and steer the appendage particles
  function updateRig() {
    if (dragon.activeWeight <= 0.001 || dragon.members.length < 6) {
      for (let i = 0; i < dragon.rig.length; i++) {
        dragon.rig[i].p.isDragonMember = false;
        dragon.rig[i].p.dragonWeight = 0;
      }
      dragon.rig = [];
      return;
    }

    if (!dragonCanHunt()) {
      for (const entry of dragon.rig) {
        entry.p.dragonWeight = Math.max(0, entry.p.dragonWeight - 0.035);
        if (entry.p.dragonWeight <= 0.001) entry.p.isDragonMember = false;
      }
      dragon.rig = dragon.rig.filter(entry => entry.p.isDragonMember);
      return;
    }

    const taken = {};
    for (let i = 0; i < dragon.rig.length; i++) taken[dragon.rig[i].slot.key] = true;

    // One new appendage particle at a time, always the nearest free one to the
    // slot, so they visibly stream outward from the body instead of appearing.
    dragon.rigCooldown--;
    if (dragon.rig.length < RIG_SLOTS.length && dragon.rigCooldown <= 0) {
      for (let s = 0; s < RIG_SLOTS.length; s++) {
        const slot = RIG_SLOTS[s];
        if (!slot.key) slot.key = slot.kind + s;
        if (taken[slot.key]) continue;
        const target = rigSlotTarget(slot);
        if (!target) continue;

        let best = null;
        let bestD = Infinity;
        for (let i = 0; i < particles.length; i++) {
          const p = particles[i];
          if (p.isDragonMember || !onArticleFlightSide(p)) continue;
          const d = Math.hypot(p.x - target.x, p.y - target.y);
          if (d < bestD) {
            bestD = d;
            best = p;
          }
        }
        if (best) {
          best.isDragonMember = true;
          best.dragonWeight = 0.06;
          best.memberIndex = -1; // appendage, not a vertebra
          dragon.rig.push({ p: best, slot: slot, settle: 0 });
          dragon.rigCooldown = 9;
        }
        break;
      }
    }

    // Pull each rig particle toward its slot with the same easing as the spine
    for (let i = dragon.rig.length - 1; i >= 0; i--) {
      const entry = dragon.rig[i];
      const p = entry.p;
      const target = rigSlotTarget(entry.slot, entry.p);
      if (!target) continue;

      p.dragonWeight = Math.min(1, p.dragonWeight + 0.018);

      const dx = target.x - p.x;
      const dy = target.y - p.y;
      const d = Math.hypot(dx, dy);
      entry.settle = 1 - Math.min(1, d / 55); // 0 = still flying in, 1 = in place
      if (d > 0.01) {
        const acc = Math.min(2.4, d * 0.16) * p.dragonWeight * dragon.activeWeight;
        p.vx += (dx / d) * acc;
        p.vy += (dy / d) * acc;
      }
    }
  }

  function drawDragon() {
    const ctx = dragonCtx;
    const mem = dragon.members;
    if (dragon.activeWeight <= 0.02 || mem.length < 4) return;

    const A = dragon.activeWeight;
    const awakeness = 1 - dragon.sleepWeight;

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // ---- Spine: a glowing filament threaded through the vertebrae ----
    // Two passes (soft wide halo, then a bright core) instead of a solid body,
    // so the individual particles still read as particles.
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 1; i < mem.length; i++) {
        const a = mem[i - 1];
        const b = mem[i];
        const d = Math.hypot(b.renderX - a.renderX, b.renderY - a.renderY);
        // Segments still being reeled in are left as recruitment beams
        if (d > 34) continue;

        const settle = Math.min(a.dragonWeight, b.dragonWeight);
        const taper = 1 - (i / mem.length) * 0.72;
        const alpha = (pass === 0 ? 0.1 : 0.4) * settle * A;
        if (alpha < 0.012) continue;

        ctx.beginPath();
        ctx.moveTo(a.renderX, a.renderY);
        ctx.lineTo(b.renderX, b.renderY);
        ctx.strokeStyle = pass === 0 ? DRAGON.midColor : '#52524d';
        ctx.lineWidth = (pass === 0 ? 9 : 2.1) * taper;
        ctx.globalAlpha = alpha;
        ctx.stroke();
      }
    }

    // ---- Wings: struts between real particles, with a faint membrane ----
    for (let w = 0; w < WINGS.length; w++) {
      const cfg = WINGS[w];
      if (mem.length <= cfg.trail) continue;

      for (let side = -1; side <= 1; side += 2) {
        let mid = null;
        let tip = null;
        for (let r = 0; r < dragon.rig.length; r++) {
          const e = dragon.rig[r];
          if (e.slot.kind !== 'wing' || e.slot.wing !== w || e.slot.side !== side) continue;
          if (e.slot.part === 'mid') mid = e;
          else tip = e;
        }
        if (!mid || !tip) continue;

        const shoulder = mem[cfg.anchor];
        const trail = mem[cfg.trail];
        // Fade the whole wing in with how settled its two joints are
        const wf = Math.min(mid.settle || 0, tip.settle || 0) *
          Math.min(mid.p.dragonWeight, tip.p.dragonWeight) * A * awakeness;
        if (wf < 0.02) continue;

        // Membrane: a wash that thins out toward the tip, so the wing reads as
        // a veil stretched between the particles rather than a solid panel
        ctx.beginPath();
        ctx.moveTo(shoulder.renderX, shoulder.renderY);
        ctx.lineTo(mid.p.renderX, mid.p.renderY);
        ctx.lineTo(tip.p.renderX, tip.p.renderY);
        ctx.lineTo(trail.renderX, trail.renderY);
        ctx.closePath();
        const veil = ctx.createLinearGradient(shoulder.renderX, shoulder.renderY, tip.p.renderX, tip.p.renderY);
        veil.addColorStop(0, 'rgba(80, 80, 74, 0.12)');
        veil.addColorStop(1, 'rgba(80, 80, 74, 0.01)');
        ctx.fillStyle = veil;
        ctx.globalAlpha = 0.75 * wf;
        ctx.fill();

        // Bones: shoulder -> elbow -> tip, plus the trailing edge back to the body
        ctx.beginPath();
        ctx.moveTo(shoulder.renderX, shoulder.renderY);
        ctx.lineTo(mid.p.renderX, mid.p.renderY);
        ctx.lineTo(tip.p.renderX, tip.p.renderY);
        ctx.strokeStyle = DRAGON.headColor;
        ctx.lineWidth = 1.3;
        ctx.globalAlpha = 0.5 * wf;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(tip.p.renderX, tip.p.renderY);
        ctx.lineTo(trail.renderX, trail.renderY);
        ctx.moveTo(mid.p.renderX, mid.p.renderY);
        ctx.lineTo(trail.renderX, trail.renderY);
        ctx.strokeStyle = DRAGON.tailColor;
        ctx.lineWidth = 0.8;
        ctx.globalAlpha = 0.3 * wf;
        ctx.stroke();
      }
    }

    // ---- Horns + face ----
    const headP = mem[0];
    const f = memberFrame(0, true);
    let dirX = dragon.head.vx;
    let dirY = dragon.head.vy;
    const dm = Math.hypot(dirX, dirY);
    if (dm < 0.05) {
      dirX = f.tx;
      dirY = f.ty;
    } else {
      dirX /= dm;
      dirY /= dm;
    }

    let snoutE = null;
    const hornE = [];
    for (let r = 0; r < dragon.rig.length; r++) {
      const e = dragon.rig[r];
      if (e.slot.kind === 'snout') snoutE = e;
      else if (e.slot.kind === 'horn') hornE.push(e);
    }

    // Skull: horn tips brace back from the brow, and each meets the snout so
    // the three particles read as a wedge-shaped head.
    for (let i = 0; i < hornE.length; i++) {
      const e = hornE[i];
      const hf = (e.settle || 0) * e.p.dragonWeight * A;
      if (hf < 0.02) continue;
      ctx.beginPath();
      ctx.moveTo(headP.renderX, headP.renderY);
      ctx.lineTo(e.p.renderX, e.p.renderY);
      ctx.strokeStyle = DRAGON.crestColor;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.45 * hf;
      ctx.stroke();

      if (snoutE) {
        const jf = hf * (snoutE.settle || 0) * snoutE.p.dragonWeight;
        if (jf > 0.02) {
          ctx.beginPath();
          ctx.moveTo(e.p.renderX, e.p.renderY);
          ctx.lineTo(snoutE.p.renderX, snoutE.p.renderY);
          ctx.strokeStyle = '#52524d';
          ctx.lineWidth = 1.2;
          ctx.globalAlpha = 0.4 * jf;
          ctx.stroke();
        }
      }
    }

    if (snoutE) {
      const sf = (snoutE.settle || 0) * snoutE.p.dragonWeight * A;
      if (sf > 0.02) {
        ctx.beginPath();
        ctx.moveTo(headP.renderX, headP.renderY);
        ctx.lineTo(snoutE.p.renderX, snoutE.p.renderY);
        ctx.strokeStyle = '#52524d';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5 * sf;
        ctx.stroke();
      }
    }

    // ---- Tail: a short forked flick of filament off the last vertebra ----
    const tail = mem[mem.length - 1];
    const tf = memberFrame(mem.length - 1, true);
    const tailAlpha = tail.dragonWeight * A;
    if (tailAlpha > 0.02) {
      ctx.beginPath();
      for (let sd = -1; sd <= 1; sd += 2) {
        ctx.moveTo(tail.renderX, tail.renderY);
        ctx.quadraticCurveTo(
          tail.renderX - tf.tx * 9 + tf.nx * sd * 3,
          tail.renderY - tf.ty * 9 + tf.ny * sd * 3,
          tail.renderX - tf.tx * 15 + tf.nx * sd * 8,
          tail.renderY - tf.ty * 15 + tf.ny * sd * 8
        );
      }
      ctx.strokeStyle = DRAGON.tailColor;
      ctx.lineWidth = 1.4;
      ctx.globalAlpha = 0.45 * tailAlpha;
      ctx.stroke();
    }

    // Sleeping: a slow drift of breath motes off the snout
    if (dragon.sleepWeight > 0.4) {
      ctx.globalAlpha = 0.28 * A * dragon.sleepWeight;
      ctx.fillStyle = DRAGON.tailColor;
      for (let z = 0; z < 3; z++) {
        const ph = (dragon.beatPhase * 0.5 + z * 2.1) % 6.3;
        ctx.beginPath();
        ctx.arc(headP.renderX + dirX * (8 + ph * 4), headP.renderY + dirY * (8 + ph * 4) - ph * 3.5, 1.4 + z * 0.35, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function updateDragonHead(now) {
    dragon.time += 0.016;
    updateArticleFlight(now);

    if (dragonCanHunt()) {
      dragon.activeWeight = Math.min(1, dragon.activeWeight + 0.05);
    } else {
      dragon.activeWeight = Math.max(0, dragon.activeWeight - 0.035);
    }

    if (dragon.activeWeight <= 0.001) return;

    keepOutsideArticle(dragon.head);
    if (articleFlight) {
      glideBesideArticle();
      return;
    }

    // Latch pursuit until arrival so the normal figure-eight can extend beyond
    // the orbit without repeatedly switching back into chase mode.
    const dx = dragon.head.x - mouse.x;
    const dy = dragon.head.y - mouse.y;
    const dist = Math.hypot(dx, dy) || 0.001;
    const awake = dragonCanHunt();
    if (!awake || dist <= dragon.pursuitEndDistance) dragon.pursuing = false;
    else if (dist >= dragon.pursuitStartDistance) dragon.pursuing = true;

    // Reach a distant resting cursor before starting idle choreography.
    const restMs = now - lastMoveAt;
    const wantSleep = restMs > SLEEP_DELAY_MS && awake && !dragon.pursuing;
    const wantIdle = restMs > IDLE_DELAY_MS && awake && !dragon.pursuing && !wantSleep;

    // Sleep creeps in slowly but breaks instantly on the first twitch of the mouse
    dragon.sleepWeight = wantSleep
      ? Math.min(1, dragon.sleepWeight + 0.005)
      : Math.max(0, dragon.sleepWeight - 0.06);

    // Slow wingbeat clock. It drives the wings only - flight speed stays
    // constant so the glide never pulses.
    dragon.beatPhase += 0.024 * (1 - 0.75 * dragon.sleepWeight);

    // Asleep: the orbit winds down into a tight, slow coil
    const orbitR = dragon.orbitRadius * (1 - dragon.sleepWeight) + 30 * dragon.sleepWeight;
    const orbitSpd = dragon.orbitSpeed * (1 - 0.72 * dragon.sleepWeight);

    if (wantIdle) {
      if (dragon.idleWeight <= 0.0001) {
        // Size the lemniscate to the viewport, then latch onto the nearest point
        // on the curve so the transition is a pathfind, never a snap.
        dragon.figA = Math.max(120, Math.min(240, width * 0.22));
        dragon.figB = dragon.figA * 0.56;
        dragon.figTheta = figClosestTheta(
          dragon.head.x - mouse.x,
          dragon.head.y - mouse.y,
          dragon.figA,
          dragon.figB
        );

        // Latch the travel direction to whichever way along the curve agrees
        // with the head's current momentum, then never change it for this run.
        const entryTan = figTangent(dragon.figTheta, dragon.figA, dragon.figB);
        const along = entryTan.x * dragon.head.vx + entryTan.y * dragon.head.vy;
        dragon.figDir = along >= 0 ? 1 : -1;
      }
      dragon.idleWeight = Math.min(1, dragon.idleWeight + 0.01);
    } else {
      dragon.idleWeight = Math.max(0, dragon.idleWeight - 0.045);
    }

    // Vector from mouse cursor to dragon head
    const nx = dx / dist; // outward radial unit vector
    const ny = dy / dist;

    // Dynamically establish spin direction on initial approach to avoid snaps.
    // Frozen during the infinity: angular momentum about the cursor legitimately
    // flips sign on each lobe, which would otherwise reverse the whole path.
    const angMom = dx * dragon.head.vy - dy * dragon.head.vx;
    if (Math.abs(angMom) > 12 && dragon.idleWeight < 0.05) {
      dragon.spinDir = angMom >= 0 ? 1 : -1;
    }

    // Tangent unit vector for circling around the cursor
    // In screen coordinates (+y down), for clockwise (spinDir = 1): tx = -ny, ty = nx
    const s = dragon.spinDir;
    const tx = -s * ny;
    const ty = s * nx;

    // Orbital pathfinding (Lyapunov-stable pursuit-orbit spiral):
    // Radial error: distance minus desired orbit radius
    const radialError = dist - orbitR;

    // Inward/outward radial velocity:
    // Gentle radial steering to smoothly spiral in/out without sudden rushes
    const vRadial = Math.max(-2.2, Math.min(2.2, radialError * 0.038));

    // Desired composite velocity: circular orbit tangent + radial pathfinding
    let desiredVx = tx * orbitSpd - nx * vRadial;
    let desiredVy = ty * orbitSpd - ny * vRadial;

    // ---- Infinity choreography (blended in while the cursor rests) ----
    if (dragon.idleWeight > 0.0001) {
      const A = dragon.figA;
      const B = dragon.figB;
      const dir = dragon.figDir;

      // Pure path-following ("carrot chasing"): re-project the head onto the
      // curve every frame instead of racing an independent clock. No lag can
      // accumulate, so the traced shape stays a true infinity.
      const ox = dragon.head.x - mouse.x;
      const oy = dragon.head.y - mouse.y;

      // Forward-only projection. Searching backwards lets the lobe tips and the
      // self-intersection snap theta onto the branch already travelled, which
      // reverses the carrot and destroys the figure-8 mid-loop.
      let bestT = dragon.figTheta;
      let bestD = Infinity;
      for (let k = 0; k <= 14; k++) {
        const t = dragon.figTheta + k * 0.045 * dir;
        const q = figPoint(t, A, B);
        const d = (q.x - ox) * (q.x - ox) + (q.y - oy) * (q.y - oy);
        if (d < bestD) {
          bestD = d;
          bestT = t;
        }
      }
      dragon.figTheta = bestT;

      // Carrot sits a fixed arc-length ahead of the projection
      const tan = figTangent(bestT, A, B);
      const lookahead = 26;
      const carrotT = bestT + (lookahead / tan.mag) * dir;
      const c = figPoint(carrotT, A, B);

      const errX = mouse.x + c.x - dragon.head.x;
      const errY = mouse.y + c.y - dragon.head.y;
      const errMag = Math.hypot(errX, errY) || 0.001;

      const figSpd = dragon.figSpeed;
      const figVx = (errX / errMag) * figSpd;
      const figVy = (errY / errMag) * figSpd;

      // Keep theta bounded
      if (dragon.figTheta > Math.PI * 6) dragon.figTheta -= Math.PI * 6;
      if (dragon.figTheta < -Math.PI * 6) dragon.figTheta += Math.PI * 6;

      const w = dragon.idleWeight;
      desiredVx = desiredVx * (1 - w) + figVx * w;
      desiredVy = desiredVy * (1 - w) + figVy * w;
    }

    if (dragon.pursuing) {
      // Far away: aim directly at the cursor. Ease the tangential orbit back in
      // on approach, with zero slope at both ends to avoid a steering seam.
      const t = Math.max(0, Math.min(1, (dist - dragon.pursuitEndDistance) /
        (dragon.pursuitStartDistance - dragon.pursuitEndDistance)));
      const pursuit = t * t * (3 - 2 * t);
      desiredVx += (-nx * dragon.pursuitSpeed - desiredVx) * pursuit;
      desiredVy += (-ny * dragon.pursuitSpeed - desiredVy) * pursuit;
    }

    // Steering acceleration with natural momentum (gentle, unhurried)
    const steerForce = 0.045 + 0.03 * dragon.idleWeight;
    const maxHeadSpeed = (2.4 + 0.7 * dragon.idleWeight) * (1 - 0.6 * dragon.sleepWeight);
    if (articleRect && articleHover && dragonCanHunt()) {
      const route = articleRoute(dragon.head, mouse, articleRail());
      if (route && !route.direct) {
        const dx = route.point.x - dragon.head.x, dy = route.point.y - dragon.head.y;
        const d = Math.hypot(dx, dy) || 1;
        desiredVx = dx / d * dragon.pursuitSpeed;
        desiredVy = dy / d * dragon.pursuitSpeed;
      }
    }
    steerHead(desiredVx, desiredVy, steerForce, maxHeadSpeed);
  }

  function onArticleFlightSide(p) {
    return !articleFlight || (articleFlight.side < 0 ? p.x < articleRect.left : p.x > articleRect.right);
  }

  function updateDragonMembers() {
    if (dragon.activeWeight <= 0.001) {
      // Disband all members
      for (let i = 0; i < dragon.members.length; i++) {
        dragon.members[i].isDragonMember = false;
        dragon.members[i].dragonWeight = 0;
      }
      dragon.members = [];
      dragon.history = [];
      return;
    }

    const targetCount = dragonLength;
    const recruitAt = articleFlight ? dragon.head : mouse;

    if (dragonCanHunt()) {
      // 1. Initial seed: recruit the particle closest to the cursor position
      if (dragon.members.length === 0 && particles.length > 0) {
        let nearest = null;
        let minDist = Infinity;
        for (let i = 0; i < particles.length; i++) {
          const p = particles[i];
          if (!onArticleFlightSide(p)) continue;
          const d = Math.hypot(p.x - recruitAt.x, p.y - recruitAt.y);
          if (d < minDist) {
            minDist = d;
            nearest = p;
          }
        }
        if (nearest) {
          // Initialize head at nearest particle position so it doesn't jump
          dragon.head.x = nearest.x;
          dragon.head.y = nearest.y;
          dragon.head.vx = nearest.vx;
          dragon.head.vy = nearest.vy;
          dragon.turnVelocity = 0;
          dragon.history = [{ x: nearest.x, y: nearest.y }];
          nearest.isDragonMember = true;
          nearest.dragonWeight = 0.05; // Starts small, gently gathers
          dragon.members.push(nearest);
          dragon.recruitCooldown = 8; // Paced delay before recruiting next segment
        }
      }

      // 2. Stream recruitment: gradually gather the unrecruited particles closest to the cursor
      if (dragon.members.length < targetCount && particles.length > 0) {
        dragon.recruitCooldown--;
        if (dragon.recruitCooldown <= 0) {
          dragon.recruitCooldown = 7; // Paced intake: 1 particle every 7 frames (~115ms) for gradual gathering
          let candidate = null;
          let minDistToCursor = Infinity;
          for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.isDragonMember) continue;
            // Measure distance to cursor so nearby particles are gathered in order
            if (!onArticleFlightSide(p)) continue;
            const d = Math.hypot(p.x - recruitAt.x, p.y - recruitAt.y);
            if (d < minDistToCursor) {
              minDistToCursor = d;
              candidate = p;
            }
          }
          if (candidate) {
            candidate.isDragonMember = true;
            candidate.dragonWeight = 0.08; // Starts with active pull weight
            dragon.members.push(candidate);
          }
        }
      }

      // 3. Gradual weight ramp: slow, smooth blending into the dragon chain
      for (let i = 0; i < dragon.members.length; i++) {
        const p = dragon.members[i];
        p.dragonWeight = Math.min(1.0, p.dragonWeight + 0.018);
      }

      // 4. Broken link release: only release established members if mouse flicked far away
      for (let i = 1; i < dragon.members.length; i++) {
        const prev = dragon.members[i - 1];
        const curr = dragon.members[i];
        if (curr.dragonWeight > 0.85 && Math.hypot(prev.x - curr.x, prev.y - curr.y) > 280) {
          for (let k = i; k < dragon.members.length; k++) {
            dragon.members[k].isDragonMember = false;
            dragon.members[k].dragonWeight = 0;
          }
          dragon.members.splice(i);
          break;
        }
      }
    } else {
      // Mouse inactive or over card: smoothly release members back to ambient dust
      for (let i = 0; i < dragon.members.length; i++) {
        const p = dragon.members[i];
        p.dragonWeight = Math.max(0.0, p.dragonWeight - 0.035);
        if (p.dragonWeight <= 0.001) {
          p.isDragonMember = false;
        }
      }
      dragon.members = dragon.members.filter((p) => p.isDragonMember);
    }

    if (dragon.members.length === 0) return;

    // 5. Kinematics along the spine: purely leader-follower pulled by the head
    // Body vertebrae follow the exact spatial path history traced by the head.
    const count = dragon.members.length;
    const targetDist = 15.0; // Spacing along the arc of the head's trajectory

    // Head vertebra: pulled directly by the circling dragon.head anchor
    const p0 = dragon.members[0];
    const w0 = p0.dragonWeight * dragon.activeWeight;
    p0.vx += (dragon.head.vx - p0.vx) * 0.35 * w0;
    p0.vy += (dragon.head.vy - p0.vy) * 0.35 * w0;
    p0.vx += (dragon.head.x - p0.x) * 0.18 * w0;
    p0.vy += (dragon.head.y - p0.y) * 0.18 * w0;

    // Follower vertebrae: sampled at distance offsets along dragon.history
    let accumDist = 0;
    let historyIdx = 0;

    for (let i = 1; i < count; i++) {
      const curr = dragon.members[i];
      const prev = dragon.members[i - 1];
      const w = Math.max(0.25, curr.dragonWeight) * dragon.activeWeight;
      const desiredArcDist = i * targetDist;


      // Walk along head's recorded path history to find the point exactly desiredArcDist behind head
      let targetX = prev.x;
      let targetY = prev.y;
      let found = false;

      while (historyIdx < dragon.history.length - 1) {
        const pA = dragon.history[historyIdx];
        const pB = dragon.history[historyIdx + 1];
        const segLen = Math.hypot(pB.x - pA.x, pB.y - pA.y);
        if (accumDist + segLen >= desiredArcDist) {
          const ratio = segLen > 0.001 ? (desiredArcDist - accumDist) / segLen : 0;
          targetX = pA.x + (pB.x - pA.x) * ratio;
          targetY = pA.y + (pB.y - pA.y) * ratio;
          found = true;
          break;
        }
        accumDist += segLen;
        historyIdx++;
      }

      // Fallback if history hasn't accumulated desiredArcDist yet: follow predecessor directly
      if (!found) {
        const dX = curr.x - prev.x;
        const dY = curr.y - prev.y;
        const dist = Math.hypot(dX, dY) || 0.001;
        targetX = prev.x + (dX / dist) * targetDist;
        targetY = prev.y + (dY / dist) * targetDist;
      }

      // No travelling wave: the body tracks the head's traced path exactly, so
      // the dragon glides instead of slithering.

      // Smooth pulling acceleration towards the target vertebra slot - NO TELEPORTATION
      const dX = targetX - curr.x;
      const dY = targetY - curr.y;
      const distToTarget = Math.hypot(dX, dY);

      if (distToTarget > 0.01) {
        // Neck segments are deliberately looser so the head whips around corners
        const neckSlack = i <= 3 ? 0.62 : 1;
        const pullFactor = (curr.dragonWeight > 0.6 ? 0.20 : 0.12) * neckSlack;
        const pullAcc = Math.min(2.6, distToTarget * pullFactor) * w;
        curr.vx += (dX / distToTarget) * pullAcc;
        curr.vy += (dY / distToTarget) * pullAcc;
      }

      // Smooth spring linkage to predecessor once close, preventing spinal tearing without snapping
      const dToPrev = Math.hypot(prev.x - curr.x, prev.y - curr.y) || 0.001;
      const maxDist = targetDist * 1.85;
      if (curr.dragonWeight > 0.8 && dToPrev > maxDist) {
        const springForce = (dToPrev - maxDist) * 0.18 * w;
        curr.vx += ((prev.x - curr.x) / dToPrev) * springForce;
        curr.vy += ((prev.y - curr.y) / dToPrev) * springForce;
      }

      // When the pulled particle arrives near its target slot, accelerate weight integration
      if (distToTarget < 30) {
        curr.dragonWeight = Math.min(1.0, curr.dragonWeight + 0.025);
      }
    }

    // 6. Gentle wake disturbance on nearby non-member ambient particles
    for (let j = 0; j < particles.length; j++) {
      const amb = particles[j];
      if (amb.isDragonMember) continue;
      for (let s = 0; s < count; s += 4) {
        const seg = dragon.members[s];
        const wdx = amb.x - seg.x;
        const wdy = amb.y - seg.y;
        if (wdx > 40 || wdx < -40 || wdy > 40 || wdy < -40) continue;
        const wdist = Math.sqrt(wdx * wdx + wdy * wdy);
        if (wdist < 40 && wdist > 0.1) {
          const push = (1 - wdist / 40) * 0.22 * dragon.activeWeight;
          amb.vx += (wdx / wdist) * push + seg.vx * 0.06 * dragon.activeWeight;
          amb.vy += (wdy / wdist) * push + seg.vy * 0.06 * dragon.activeWeight;
        }
      }
    }
  }

  function simulate(now) {
    for (const p of particles) {
      p.previousX = p.x;
      p.previousY = p.y;
    }
    refreshArticleHover();
    updateDragonHead(now);
    updateDragonMembers();
    updateRig();
    applyCardRepulsion();
    for (let i = 0; i < dragon.members.length; i++) {
      dragon.members[i].memberIndex = i;
    }
    for (const p of particles) {
      if (p.isDragonMember) softenDragonWall(p);
      p.update();
      keepOutsideArticle(p, !p.isDragonMember);
    }
  }

  function render(alpha) {
    for (const p of particles) {
      p.renderX = p.previousX + (p.x - p.previousX) * alpha;
      p.renderY = p.previousY + (p.y - p.previousY) * alpha;
    }
    ctx.clearRect(0, 0, width, height);
    if (dragonCtx !== ctx) dragonCtx.clearRect(0, 0, width, height);
    drawConnections();
    drawDragon();
    drawParticles();
    drawParticles(true);
    dragonCtx.globalAlpha = 1;
    ctx.globalAlpha = 1;
  }

  function scheduleFrame() {
    if (animationFrameId === null && enabled && !document.hidden) {
      animationFrameId = requestAnimationFrame(loop);
    }
  }

  function loop(now) {
    animationFrameId = null;
    if (!enabled || document.hidden) return;
    if (resizeDirty) resize();
    if (activeCard) {
      if (!activeCardRect || cardRectDirty) {
        activeCardRect = activeCard.getBoundingClientRect();
        cardRectDirty = false;
      }
    } else {
      activeCardRect = null;
    }
    refreshArticleRect();
    refreshArticleHover();
    // Also resolve a changed article rectangle on frames without a physics step.
    for (const p of particles) keepOutsideArticle(p);

    if (lastFrameAt !== null) {
      // Bound work after a stall and discard old time, rather than spiralling
      // into unbounded catch-up. Ordinary 30/60/120 Hz all advance at 60 Hz.
      accumulator = Math.min(accumulator + Math.max(0, now - lastFrameAt), STEP_MS * MAX_STEPS);
      let steps = 0;
      while (accumulator + 1e-7 >= STEP_MS && steps < MAX_STEPS) {
        simulate(now - accumulator + STEP_MS);
        accumulator = Math.max(0, accumulator - STEP_MS);
        steps++;
      }
    }
    lastFrameAt = now;
    render(accumulator / STEP_MS);
    scheduleFrame();
  }

  function start() {
    if (!enabled || document.hidden || animationFrameId !== null) return;
    lastFrameAt = null;
    accumulator = 0;
    resizeDirty = true;
    cardRectDirty = articleRectDirty = true;
    for (const p of particles) {
      p.previousX = p.x;
      p.previousY = p.y;
    }
    scheduleFrame();
  }

  function stop(clear = true) {
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
    lastFrameAt = null;
    accumulator = 0;
    if (clear) {
      ctx.clearRect(0, 0, width, height);
      if (dragonCtx !== ctx) dragonCtx.clearRect(0, 0, width, height);
    }
  }

  // Pointer & Gravity Listeners
  function onPointerMove(e) {
    if (!enabled) return;
    const moved = Math.abs(e.clientX - mouse.x) + Math.abs(e.clientY - mouse.y);
    if (moved > 2) {
      lastMoveAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    }
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    refreshArticleRect();
    mouse.active = true;
    refreshArticleHover();
    if (!isOverArticle()) {
      articleFlight = null;
      readingDragonArmed = true;
    }
    if (!mouse.hasMoved) {
      dragon.head.x = e.clientX;
      dragon.head.y = e.clientY;
      mouse.hasMoved = true;
    }
    mouse.active = true;
  }

  function onPointerDown() {
    if (!enabled) return;
    lastMoveAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  }

  // Attach hover gravity to interactive cards and elements
  function setupInteractiveHooks() {
    // 0. Header exclusion: mouse over header deactivates dragon
    const header = document.querySelector('.wrapper-masthead');
    if (header) {
      header.addEventListener('mouseenter', () => {
        isHeaderHovered = true;
      });
      header.addEventListener('mouseleave', () => {
        isHeaderHovered = false;
      });
    }

    // 1. Post cards: trigger whole-screen slow pull + card blocking + perimeter orbit
    const cards = document.querySelectorAll('.post-card');
    cards.forEach((card) => {
      card.addEventListener('mouseenter', () => {
        activeCard = card;
        activeCardRect = card.getBoundingClientRect();
        cardRectDirty = false;
      });

      card.addEventListener('mouseleave', (e) => {
        if (!card.contains(e.relatedTarget)) {
          if (activeCard === card) {
            activeCard = null;
            activeCardRect = null;
          }
        }
      });
    });

    // 2. Navigation and button hooks (when not over a post card)
    const elements = document.querySelectorAll('.site-nav a, .post-back-link, .post-nav-card');
    elements.forEach((el) => {
      el.addEventListener('mouseenter', (e) => {
        if (!activeCard) {
          pointAttractor.active = true;
          pointAttractor.x = e.clientX;
          pointAttractor.y = e.clientY;
        }
      });

      el.addEventListener('mousemove', (e) => {
        if (!activeCard && pointAttractor.active) {
          pointAttractor.x = e.clientX;
          pointAttractor.y = e.clientY;
        }
      });

      el.addEventListener('mouseleave', () => {
        if (!activeCard) {
          pointAttractor.active = false;
        }
      });
    });
  }

  // Pause while hidden; resuming never simulates the time spent away.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop(false);
    else start();
  });
  window.addEventListener('pagehide', () => stop(false));
  window.addEventListener('pageshow', start);
  motionPreference.addEventListener('change', () => {
    enabled = !motionPreference.matches;
    if (enabled) start();
    else stop();
  });

  window.addEventListener('resize', () => {
    resizeDirty = true;
    cardRectDirty = articleRectDirty = true;
  });

  window.addEventListener('scroll', () => {
    cardRectDirty = true;
    articleRectDirty = true;
  }, { passive: true });

  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerdown', onPointerDown, { passive: true });
  document.addEventListener('mouseleave', () => {
    mouse.active = false;
  });

  function init() {
    setupInteractiveHooks();
    const modeBtn = document.getElementById('particle-mode');
    if (modeBtn) {
      syncModeSwitch();
      modeBtn.addEventListener('click', (e) => {
        const opt = e.target.closest('.particle-mode-opt');
        if (opt && opt.getAttribute('data-mode')) {
          setAmbientMode(opt.getAttribute('data-mode'));
        } else {
          setAmbientMode(ambientMode === 'web' ? 'flow' : 'web');
        }
      });
    }
    if (enabled) start();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
