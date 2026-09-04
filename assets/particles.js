/**
 * Interactive Cosmic Gravity & Constellation Canvas Engine
 * Zero dependencies, battery-conscious, high-performance particle background.
 *
 * Behaviors:
 * 1. Background Cursor: Living Cosmic Serpent dynamically recruited from nearby ambient particles.
 *    Smoothly pathfinds via a pursuit-orbit spiral into a stable, fluid circular orbit around
 *    the cursor. Particles are dragged/flowed into the chain with bilateral spring tension,
 *    momentum transfer, and transverse undulation waves.
 * 2. Blog Post Card Hover: Pure rectangular gravity, solid box blocking & perimeter orbit.
 * 3. Ambient Dust & Idle: Fluid wake effects when serpent slithers past stars; graceful release
 *    and fast reset to neutral Brownian drift when idle.
 * 4. Constellation Filaments: Proximity linkages web between serpent coils and background stars.
 */
(function () {
  'use strict';

  // Check user preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const storedSetting = localStorage.getItem('particles_enabled');
  let enabled = storedSetting !== null ? storedSetting === 'true' : !prefersReducedMotion;

  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let animationFrameId = null;
  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);

  // Palette: Subtle cosmic luminescence
  const PALETTE = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#60a5fa'];

  // Particle configuration
  const config = {
    count: window.innerWidth < 640 ? 60 : 120,
    maxDistance: 95,
    damping: 0.965,
    ambientSpeed: 0.45
  };

  // State for active card and header
  let activeCard = null;
  let activeCardRect = null;
  let isHeaderHovered = false;

  // Mouse & Cosmic Serpent Engine
  const mouse = {
    x: 0,
    y: 0,
    active: false,
    hasMoved: false
  };

  let snakeLength = window.innerWidth < 640 ? 14 : 20;

  const snake = {
    head: { x: 0, y: 0, vx: 0, vy: 0 },
    history: [], // [{x, y}], stores exact path samples recorded by the head
    members: [], // Dynamically recruited Particle instances [P_head, ..., P_tail]
    orbitRadius: 78,
    orbitSpeed: 1.6, // Relaxed, tranquil orbit speed
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
    figDir: 1 // travel direction along the curve, latched for the whole idle run
  };

  const IDLE_DELAY_MS = 3800; // let it circle a while before the infinity kicks in
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

  // Shockwave array for clicks
  const shockwaves = [];

  class Particle {
    constructor(index = 0, total = 100) {
      this.index = index;
      this.total = total;
      this.isSnakeMember = false;
      this.snakeWeight = 0;
      this.reset(true);
    }

    reset(initial = false) {
      this.x = Math.random() * width;
      this.y = initial ? Math.random() * height : (Math.random() < 0.5 ? -10 : height + 10);

      const angle = Math.random() * Math.PI * 2;
      const speed = (0.2 + Math.random() * 0.35) * config.ambientSpeed;
      this.baseVx = Math.cos(angle) * speed;
      this.baseVy = Math.sin(angle) * speed;

      this.vx = this.baseVx;
      this.vy = this.baseVy;

      this.radius = Math.random() * 1.5 + 1.1;
      this.color = PALETTE[Math.floor(Math.random() * PALETTE.length)];
      this.alpha = Math.random() * 0.45 + 0.4;
      this.orbitOffset = (Math.random() - 0.5) * 22; // Particle-specific orbital track
      this.orbitSpeedFactor = 0.82 + Math.random() * 0.36;

      this.isSnakeMember = false;
      this.snakeWeight = 0;
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
        const distToBox = Math.hypot(dx, dy);

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
        const dist = Math.hypot(dx, dy) || 1;
        const pull = 0.055;
        this.vx += (dx / dist) * pull;
        this.vy += (dy / dist) * pull;
      }

      // 3. Shockwave impulses from user clicks
      for (let i = 0; i < shockwaves.length; i++) {
        const wave = shockwaves[i];
        const dx = this.x - wave.x;
        const dy = this.y - wave.y;
        const dist = Math.hypot(dx, dy) || 1;
        const diff = dist - wave.currentRadius;

        if (Math.abs(diff) < 35) {
          const pushForce = (1 - wave.currentRadius / wave.maxRadius) * 2.2;
          this.vx += (dx / dist) * pushForce;
          this.vy += (dy / dist) * pushForce;
        }
      }

      // 4. Speed limiter & damping per mode
      if (activeCard) {
        // Blog post card hover - keep exact current good speed and fluid damping
        const maxSpeed = 3.4;
        const currentSpeed = Math.hypot(this.vx, this.vy);
        if (currentSpeed > maxSpeed) {
          this.vx = (this.vx / currentSpeed) * maxSpeed;
          this.vy = (this.vy / currentSpeed) * maxSpeed;
        }
        const currentDamping = 0.978;
        this.vx = this.vx * currentDamping + this.baseVx * (1 - currentDamping);
        this.vy = this.vy * currentDamping + this.baseVy * (1 - currentDamping);
      } else if (this.isSnakeMember) {
        // Dynamically recruited serpent member: fluid damping & speed cap
        const maxSpeed = 3.2;
        const currentSpeed = Math.hypot(this.vx, this.vy);
        if (currentSpeed > maxSpeed) {
          this.vx = (this.vx / currentSpeed) * maxSpeed;
          this.vy = (this.vy / currentSpeed) * maxSpeed;
        }
        const w = Math.min(1.0, Math.max(0.2, this.snakeWeight));
        const snakeDamping = 0.92;
        const ambientDamping = 0.88;
        const effDamping = ambientDamping * (1 - w) + snakeDamping * w;
        // Pure velocity decay for snake members so ambient drift doesn't bias the trailing vertebrae
        this.vx = this.vx * effDamping + this.baseVx * 0.04 * (1 - w);
        this.vy = this.vy * effDamping + this.baseVy * 0.04 * (1 - w);
      } else {
        // Neutral chaos state / Ambient stars: Rapidly reset velocity back to ambient Brownian drift
        const resetSpeed = 0.12;
        this.vx += (this.baseVx - this.vx) * resetSpeed;
        this.vy += (this.baseVy - this.vy) * resetSpeed;

        // Subtle organic wander so neutral particles drift naturally in all directions
        this.baseVx += (Math.random() - 0.5) * 0.03;
        this.baseVy += (Math.random() - 0.5) * 0.03;
        const speed = Math.hypot(this.baseVx, this.baseVy);
        const targetSpeed = 0.42;
        if (speed > 0.001) {
          this.baseVx = (this.baseVx / speed) * targetSpeed;
          this.baseVy = (this.baseVy / speed) * targetSpeed;
        }
      }

      this.x += this.vx;
      this.y += this.vy;

      // 5. Viewport boundary wrapping
      if (this.x < -25) this.x = width + 25;
      if (this.x > width + 25) this.x = -25;
      if (this.y < -25) this.y = height + 25;
      if (this.y > height + 25) this.y = -25;
    }

    draw() {
      // Glow aura around dynamically recruited snake head
      if (snake.members.length > 0 && this === snake.members[0] && this.snakeWeight > 0.1) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 2.8, 0, Math.PI * 2);
        ctx.fillStyle = '#818cf8';
        ctx.globalAlpha = 0.35 * this.snakeWeight * snake.activeWeight;
        ctx.fill();
        ctx.restore();
      }

      ctx.beginPath();
      let rad = this.radius;
      if (this.isSnakeMember && this.snakeWeight > 0.05) {
        const memberIdx = snake.members.indexOf(this);
        const ratio = memberIdx >= 0 ? memberIdx / Math.max(1, snake.members.length) : 0.5;
        const targetRad = this.radius * (1.35 - 0.45 * ratio);
        rad = this.radius * (1 - this.snakeWeight) + targetRad * this.snakeWeight;
      }
      ctx.arc(this.x, this.y, rad, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.alpha;
      ctx.fill();
    }
  }

  let particles = [];

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    ctx.scale(dpr, dpr);

    snakeLength = width < 640 ? 14 : 20;
    const targetCount = width < 640 ? 60 : 120;
    if (particles.length !== targetCount) {
      snake.members = [];
      snake.history = [];
      particles = Array.from({ length: targetCount }, (_, i) => new Particle(i, targetCount));
    }
  }

  function drawConnections() {
    // 1. Glowing connected snake spine across dynamically recruited members
    if (snake.members.length > 1) {
      ctx.save();
      for (let i = 1; i < snake.members.length; i++) {
        const p1 = snake.members[i - 1];
        const p2 = snake.members[i];
        const d = Math.hypot(p2.x - p1.x, p2.y - p1.y);

        if (d < 36) {
          // Connected spine link between adjacent vertebrae
          const segAlpha = 0.45 * Math.min(p1.snakeWeight, p2.snakeWeight) * snake.activeWeight * (1 - d / 40);
          if (segAlpha > 0.01) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = '#a5b4fc';
            ctx.lineWidth = 1.9;
            ctx.globalAlpha = segAlpha;
            ctx.stroke();
          }
        } else if (snake.activeWeight > 0.05) {
          // Being pulled from ambient space toward the serpent: subtle magnetic attraction beam
          const pullBeamAlpha = 0.16 * Math.max(0.1, 1 - d / 300) * snake.activeWeight;
          if (pullBeamAlpha > 0.01) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = '#818cf8';
            ctx.lineWidth = 0.85;
            ctx.globalAlpha = pullBeamAlpha;
            ctx.stroke();
          }
        }
      }
      ctx.restore();
    }

    // 2. Pairwise proximity constellation lines
    const maxDist = config.maxDistance;
    const maxDistSq = maxDist * maxDist;
    const len = particles.length;

    ctx.lineWidth = 0.75;

    for (let i = 0; i < len; i++) {
      const p1 = particles[i];
      for (let j = i + 1; j < len; j++) {
        const p2 = particles[j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const distSq = dx * dx + dy * dy;

        if (distSq < maxDistSq) {
          const dist = Math.sqrt(distSq);

          // Anti-clumping repulsion: ONLY active when hovering over a blog post card!
          if (activeCard && dist < 24 && dist > 0.001) {
            const repelStrength = (1 - dist / 24) * 0.38;
            const rx = (dx / dist) * repelStrength;
            const ry = (dy / dist) * repelStrength;
            p1.vx += rx;
            p1.vy += ry;
            p2.vx -= rx;
            p2.vy -= ry;
          }

          const alpha = (1 - dist / maxDist) * 0.20;
          ctx.strokeStyle = '#818cf8';
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }
  }

  function updateShockwaves() {
    for (let i = shockwaves.length - 1; i >= 0; i--) {
      const wave = shockwaves[i];
      wave.currentRadius += 7;
      if (wave.currentRadius >= wave.maxRadius) {
        shockwaves.splice(i, 1);
      }
    }
  }

  function updateSnakeHead() {
    snake.time += 0.016;

    if (mouse.active && !activeCard && !isHeaderHovered) {
      snake.activeWeight = Math.min(1, snake.activeWeight + 0.05);
    } else {
      snake.activeWeight = Math.max(0, snake.activeWeight - 0.035);
    }

    if (snake.activeWeight <= 0.001) return;

    // ---- Idle detection: after a rest period, choreograph a big infinity ----
    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const wantIdle = (now - lastMoveAt) > IDLE_DELAY_MS && mouse.active && !activeCard && !isHeaderHovered;

    if (wantIdle) {
      if (snake.idleWeight <= 0.0001) {
        // Size the lemniscate to the viewport, then latch onto the nearest point
        // on the curve so the transition is a pathfind, never a snap.
        snake.figA = Math.max(120, Math.min(240, width * 0.22));
        snake.figB = snake.figA * 0.56;
        snake.figTheta = figClosestTheta(
          snake.head.x - mouse.x,
          snake.head.y - mouse.y,
          snake.figA,
          snake.figB
        );

        // Latch the travel direction to whichever way along the curve agrees
        // with the head's current momentum, then never change it for this run.
        const entryTan = figTangent(snake.figTheta, snake.figA, snake.figB);
        const along = entryTan.x * snake.head.vx + entryTan.y * snake.head.vy;
        snake.figDir = along >= 0 ? 1 : -1;
      }
      snake.idleWeight = Math.min(1, snake.idleWeight + 0.01);
    } else {
      snake.idleWeight = Math.max(0, snake.idleWeight - 0.045);
    }

    // Vector from mouse cursor to snake head
    const dx = snake.head.x - mouse.x;
    const dy = snake.head.y - mouse.y;
    const dist = Math.hypot(dx, dy) || 0.001;
    const nx = dx / dist; // outward radial unit vector
    const ny = dy / dist;

    // Dynamically establish spin direction on initial approach to avoid snaps.
    // Frozen during the infinity: angular momentum about the cursor legitimately
    // flips sign on each lobe, which would otherwise reverse the whole path.
    const angMom = dx * snake.head.vy - dy * snake.head.vx;
    if (Math.abs(angMom) > 12 && snake.idleWeight < 0.05) {
      snake.spinDir = angMom >= 0 ? 1 : -1;
    }

    // Tangent unit vector for circling around the cursor
    // In screen coordinates (+y down), for clockwise (spinDir = 1): tx = -ny, ty = nx
    const s = snake.spinDir;
    const tx = -s * ny;
    const ty = s * nx;

    // Orbital pathfinding (Lyapunov-stable pursuit-orbit spiral):
    // Radial error: distance minus desired orbit radius
    const radialError = dist - snake.orbitRadius;

    // Inward/outward radial velocity:
    // Gentle radial steering to smoothly spiral in/out without sudden rushes
    const vRadial = Math.max(-2.2, Math.min(2.2, radialError * 0.038));

    // Desired composite velocity: circular orbit tangent + radial pathfinding
    let desiredVx = tx * snake.orbitSpeed - nx * vRadial;
    let desiredVy = ty * snake.orbitSpeed - ny * vRadial;

    // ---- Infinity choreography (blended in while the cursor rests) ----
    if (snake.idleWeight > 0.0001) {
      const A = snake.figA;
      const B = snake.figB;
      const dir = snake.figDir;

      // Pure path-following ("carrot chasing"): re-project the head onto the
      // curve every frame instead of racing an independent clock. No lag can
      // accumulate, so the traced shape stays a true infinity.
      const ox = snake.head.x - mouse.x;
      const oy = snake.head.y - mouse.y;

      // Forward-only projection. Searching backwards lets the lobe tips and the
      // self-intersection snap theta onto the branch already travelled, which
      // reverses the carrot and destroys the figure-8 mid-loop.
      let bestT = snake.figTheta;
      let bestD = Infinity;
      for (let k = 0; k <= 14; k++) {
        const t = snake.figTheta + k * 0.045 * dir;
        const q = figPoint(t, A, B);
        const d = (q.x - ox) * (q.x - ox) + (q.y - oy) * (q.y - oy);
        if (d < bestD) {
          bestD = d;
          bestT = t;
        }
      }
      snake.figTheta = bestT;

      // Carrot sits a fixed arc-length ahead of the projection
      const tan = figTangent(bestT, A, B);
      const lookahead = 26;
      const carrotT = bestT + (lookahead / tan.mag) * dir;
      const c = figPoint(carrotT, A, B);

      const errX = mouse.x + c.x - snake.head.x;
      const errY = mouse.y + c.y - snake.head.y;
      const errMag = Math.hypot(errX, errY) || 0.001;

      const figVx = (errX / errMag) * snake.figSpeed;
      const figVy = (errY / errMag) * snake.figSpeed;

      // Keep theta bounded
      if (snake.figTheta > Math.PI * 6) snake.figTheta -= Math.PI * 6;
      if (snake.figTheta < -Math.PI * 6) snake.figTheta += Math.PI * 6;

      const w = snake.idleWeight;
      desiredVx = desiredVx * (1 - w) + figVx * w;
      desiredVy = desiredVy * (1 - w) + figVy * w;
    }

    // Steering acceleration with natural momentum (gentle, unhurried)
    const steerForce = 0.045 + 0.03 * snake.idleWeight;
    snake.head.vx += (desiredVx - snake.head.vx) * steerForce;
    snake.head.vy += (desiredVy - snake.head.vy) * steerForce;

    // Speed limiter on head: keep relaxed and graceful
    const maxHeadSpeed = 2.4 + 0.7 * snake.idleWeight;
    const headSpeed = Math.hypot(snake.head.vx, snake.head.vy);
    if (headSpeed > maxHeadSpeed) {
      snake.head.vx = (snake.head.vx / headSpeed) * maxHeadSpeed;
      snake.head.vy = (snake.head.vy / headSpeed) * maxHeadSpeed;
    }

    snake.head.x += snake.head.vx;
    snake.head.y += snake.head.vy;

    // Record high-resolution breadcrumb trail for snake body vertebrae
    snake.history.unshift({ x: snake.head.x, y: snake.head.y });
    const maxHistorySamples = (snakeLength + 2) * 8;
    if (snake.history.length > maxHistorySamples) {
      snake.history.length = maxHistorySamples;
    }
  }

  function updateSnakeMembers() {
    if (snake.activeWeight <= 0.001) {
      // Disband all members
      for (let i = 0; i < snake.members.length; i++) {
        snake.members[i].isSnakeMember = false;
        snake.members[i].snakeWeight = 0;
      }
      snake.members = [];
      snake.history = [];
      return;
    }

    const targetCount = snakeLength;

    if (mouse.active && !activeCard && !isHeaderHovered) {
      // 1. Initial seed: recruit the particle closest to the cursor position
      if (snake.members.length === 0 && particles.length > 0) {
        let nearest = null;
        let minDist = Infinity;
        for (let i = 0; i < particles.length; i++) {
          const p = particles[i];
          const d = Math.hypot(p.x - mouse.x, p.y - mouse.y);
          if (d < minDist) {
            minDist = d;
            nearest = p;
          }
        }
        if (nearest) {
          // Initialize head at nearest particle position so it doesn't jump
          snake.head.x = nearest.x;
          snake.head.y = nearest.y;
          snake.head.vx = nearest.vx;
          snake.head.vy = nearest.vy;
          snake.history = [{ x: nearest.x, y: nearest.y }];
          nearest.isSnakeMember = true;
          nearest.snakeWeight = 0.05; // Starts small, gently gathers
          snake.members.push(nearest);
          snake.recruitCooldown = 8; // Paced delay before recruiting next segment
        }
      }

      // 2. Stream recruitment: gradually gather the unrecruited particles closest to the cursor
      if (snake.members.length < targetCount && particles.length > 0) {
        snake.recruitCooldown--;
        if (snake.recruitCooldown <= 0) {
          snake.recruitCooldown = 7; // Paced intake: 1 particle every 7 frames (~115ms) for gradual gathering
          let candidate = null;
          let minDistToCursor = Infinity;
          for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.isSnakeMember) continue;
            // Measure distance to cursor so nearby particles are gathered in order
            const d = Math.hypot(p.x - mouse.x, p.y - mouse.y);
            if (d < minDistToCursor) {
              minDistToCursor = d;
              candidate = p;
            }
          }
          if (candidate) {
            candidate.isSnakeMember = true;
            candidate.snakeWeight = 0.08; // Starts with active pull weight
            snake.members.push(candidate);
          }
        }
      }

      // 3. Gradual weight ramp: slow, smooth blending into the snake chain
      for (let i = 0; i < snake.members.length; i++) {
        const p = snake.members[i];
        p.snakeWeight = Math.min(1.0, p.snakeWeight + 0.018);
      }

      // 4. Broken link release: only release established members if mouse flicked far away
      for (let i = 1; i < snake.members.length; i++) {
        const prev = snake.members[i - 1];
        const curr = snake.members[i];
        if (curr.snakeWeight > 0.85 && Math.hypot(prev.x - curr.x, prev.y - curr.y) > 280) {
          for (let k = i; k < snake.members.length; k++) {
            snake.members[k].isSnakeMember = false;
            snake.members[k].snakeWeight = 0;
          }
          snake.members.splice(i);
          break;
        }
      }
    } else {
      // Mouse inactive or over card: smoothly release members back to ambient dust
      for (let i = 0; i < snake.members.length; i++) {
        const p = snake.members[i];
        p.snakeWeight = Math.max(0.0, p.snakeWeight - 0.035);
        if (p.snakeWeight <= 0.001) {
          p.isSnakeMember = false;
        }
      }
      snake.members = snake.members.filter((p) => p.isSnakeMember);
    }

    if (snake.members.length === 0) return;

    // 5. Kinematics along the spine: purely leader-follower pulled by the head
    // Body vertebrae follow the exact spatial path history traced by the head.
    const count = snake.members.length;
    const targetDist = 13.0; // Spacing along the arc of the head's trajectory

    // Head vertebra: pulled directly by the circling snake.head anchor
    const p0 = snake.members[0];
    const w0 = p0.snakeWeight * snake.activeWeight;
    p0.vx += (snake.head.vx - p0.vx) * 0.35 * w0;
    p0.vy += (snake.head.vy - p0.vy) * 0.35 * w0;
    p0.vx += (snake.head.x - p0.x) * 0.18 * w0;
    p0.vy += (snake.head.y - p0.y) * 0.18 * w0;

    // Follower vertebrae: sampled at distance offsets along snake.history
    let accumDist = 0;
    let historyIdx = 0;

    for (let i = 1; i < count; i++) {
      const curr = snake.members[i];
      const prev = snake.members[i - 1];
      const w = Math.max(0.25, curr.snakeWeight) * snake.activeWeight;
      const desiredArcDist = i * targetDist;

      // Walk along head's recorded path history to find the point exactly desiredArcDist behind head
      let targetX = prev.x;
      let targetY = prev.y;
      let found = false;

      while (historyIdx < snake.history.length - 1) {
        const pA = snake.history[historyIdx];
        const pB = snake.history[historyIdx + 1];
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

      // Smooth pulling acceleration towards the target vertebra slot - NO TELEPORTATION
      const dX = targetX - curr.x;
      const dY = targetY - curr.y;
      const distToTarget = Math.hypot(dX, dY);

      if (distToTarget > 0.01) {
        const pullFactor = curr.snakeWeight > 0.6 ? 0.20 : 0.12;
        const pullAcc = Math.min(2.6, distToTarget * pullFactor) * w;
        curr.vx += (dX / distToTarget) * pullAcc;
        curr.vy += (dY / distToTarget) * pullAcc;
      }

      // Smooth spring linkage to predecessor once close, preventing spinal tearing without snapping
      const dToPrev = Math.hypot(prev.x - curr.x, prev.y - curr.y) || 0.001;
      const maxDist = targetDist * 1.85;
      if (curr.snakeWeight > 0.8 && dToPrev > maxDist) {
        const springForce = (dToPrev - maxDist) * 0.18 * w;
        curr.vx += ((prev.x - curr.x) / dToPrev) * springForce;
        curr.vy += ((prev.y - curr.y) / dToPrev) * springForce;
      }

      // When the pulled particle arrives near its target slot, accelerate weight integration
      if (distToTarget < 30) {
        curr.snakeWeight = Math.min(1.0, curr.snakeWeight + 0.025);
      }
    }

    // 6. Gentle wake disturbance on nearby non-member ambient particles
    for (let j = 0; j < particles.length; j++) {
      const amb = particles[j];
      if (amb.isSnakeMember) continue;
      for (let s = 0; s < count; s += 4) {
        const seg = snake.members[s];
        const wdx = amb.x - seg.x;
        const wdy = amb.y - seg.y;
        const wdist = Math.hypot(wdx, wdy);
        if (wdist < 40 && wdist > 0.1) {
          const push = (1 - wdist / 40) * 0.22 * snake.activeWeight;
          amb.vx += (wdx / wdist) * push + seg.vx * 0.06 * snake.activeWeight;
          amb.vy += (wdy / wdist) * push + seg.vy * 0.06 * snake.activeWeight;
        }
      }
    }
  }

  function loop() {
    if (!enabled) return;

    // Cache current card bounding rect for high-accuracy animation during scroll/render
    if (activeCard) {
      activeCardRect = activeCard.getBoundingClientRect();
    } else {
      activeCardRect = null;
    }

    // Update the Cosmic Serpent physics (orbital head pathfinding & dynamic member kinematics)
    updateSnakeHead();
    updateSnakeMembers();

    ctx.clearRect(0, 0, width, height);

    drawConnections();
    updateShockwaves();

    for (let i = 0; i < particles.length; i++) {
      particles[i].update();
      particles[i].draw();
    }

    ctx.globalAlpha = 1.0;
    animationFrameId = requestAnimationFrame(loop);
  }

  function start() {
    if (!animationFrameId && enabled) {
      resize();
      const targetCount = width < 640 ? 60 : 120;
      particles = Array.from({ length: targetCount }, (_, i) => new Particle(i, targetCount));
      snake.head.x = width / 2;
      snake.head.y = height / 2;
      animationFrameId = requestAnimationFrame(loop);
    }
  }

  function stop() {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    if (ctx) {
      ctx.clearRect(0, 0, width, height);
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
    if (!mouse.hasMoved) {
      snake.head.x = e.clientX;
      snake.head.y = e.clientY;
      mouse.hasMoved = true;
    }
    mouse.active = true;
  }

  function onPointerDown(e) {
    if (!enabled) return;
    lastMoveAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    shockwaves.push({
      x: e.clientX,
      y: e.clientY,
      currentRadius: 10,
      maxRadius: 180
    });
  }

  // Attach hover gravity to interactive cards and elements
  function setupInteractiveHooks() {
    // 0. Header exclusion: mouse over header deactivates snake
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
    const elements = document.querySelectorAll('.site-nav a, .site-nav button, .post-back-link, .post-nav-card');
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

  // Visibility changes (battery savings when switching tabs)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
    } else if (enabled) {
      animationFrameId = requestAnimationFrame(loop);
    }
  });

  window.addEventListener('resize', () => {
    if (enabled) {
      resize();
      if (activeCard) {
        activeCardRect = activeCard.getBoundingClientRect();
      }
    }
  });

  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerdown', onPointerDown, { passive: true });
  document.addEventListener('mouseleave', () => {
    mouse.active = false;
  });

  // Toggle API for button control
  window.toggleParticles = function (forceState) {
    enabled = typeof forceState === 'boolean' ? forceState : !enabled;
    localStorage.setItem('particles_enabled', enabled ? 'true' : 'false');
    const btn = document.getElementById('particle-toggle');
    if (btn) {
      btn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
      btn.classList.toggle('active', enabled);
      btn.innerHTML = enabled ? '<span class="icon">&#9879;</span> FX: ON' : '<span class="icon">&#9879;</span> FX: OFF';
    }
    if (enabled) {
      start();
    } else {
      stop();
    }
    return enabled;
  };

  // Lifecycle initialization
  function init() {
    setupInteractiveHooks();
    const toggleBtn = document.getElementById('particle-toggle');
    if (toggleBtn) {
      toggleBtn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
      toggleBtn.classList.toggle('active', enabled);
      toggleBtn.innerHTML = enabled ? '<span class="icon">&#9879;</span> FX: ON' : '<span class="icon">&#9879;</span> FX: OFF';
      toggleBtn.addEventListener('click', () => window.toggleParticles());
    }
    if (enabled) start();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
