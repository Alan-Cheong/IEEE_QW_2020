/* Relativistic Inner Planets Simulator (1PN around the Sun)
   - Simulates Mercury, Venus, Earth, Mars orbiting the Sun
   - Uses 1PN (first post-Newtonian) central-force correction for Schwarzschild field
   - Integrator: RK4
   - Units: SI (meters, seconds)
*/

(() => {
  'use strict';

  // Constants (SI)
  const G = 6.67430e-11; // m^3 kg^-1 s^-2
  const M_SUN = 1.98847e30; // kg
  const MU = G * M_SUN; // GM of Sun
  const C = 299792458; // m/s
  const AU = 1.495978707e11; // m
  const HOURS = 3600; // s
  const DAYS = 86400; // s
  const YEARS = 365.25 * DAYS; // s

  // Simulation parameters
  const BASE_DT = 1 * HOURS; // s per integration substep

  // Canvas & UI
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');
  const playPauseBtn = document.getElementById('playPause');
  const resetBtn = document.getElementById('reset');
  const relativityToggle = document.getElementById('relativityToggle');
  const speedSlider = document.getElementById('speedSlider');
  const speedValue = document.getElementById('speedValue');
  const trailSlider = document.getElementById('trailSlider');
  const trailValue = document.getElementById('trailValue');
  const scaleSelect = document.getElementById('scaleSelect');
  const timeLabel = document.getElementById('timeLabel');
  const modeLabel = document.getElementById('modeLabel');

  let running = true;
  let stepsPerFrame = Number(speedSlider.value);
  let trailMax = Number(trailSlider.value);

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Utility vector math
  function dot(ax, ay, bx, by) { return ax * bx + ay * by; }
  function norm(ax, ay) { return Math.hypot(ax, ay); }

  // 1PN central acceleration around the Sun
  function acceleration1PN(x, y, vx, vy) {
    const r2 = x * x + y * y;
    const r = Math.sqrt(r2);
    const inv_r3 = 1 / (r2 * r);

    // Newtonian
    const aNx = -MU * x * inv_r3;
    const aNy = -MU * y * inv_r3;

    if (!relativityToggle.checked) return [aNx, aNy];

    // 1PN correction for test particle in Schwarzschild field (harmonic/isotropic coordinates)
    // a = -mu r / r^3 + (mu / c^2 r^3) [ (4 mu/r - v^2) r + 4 (r·v) v ]
    const v2 = vx * vx + vy * vy;
    const rv = x * vx + y * vy;
    const factor = MU / (C * C) * inv_r3;
    const pnTerm = (4 * MU / r - v2);
    const aPNx = factor * (pnTerm * x + 4 * rv * vx);
    const aPNy = factor * (pnTerm * y + 4 * rv * vy);

    return [aNx + aPNx, aNy + aPNy];
  }

  // RK4 for 2D position-velocity under acceleration(x,y,vx,vy)
  function rk4Step(body, dt) {
    const { x, y, vx, vy } = body;

    // k1
    const [ax1, ay1] = acceleration1PN(x, y, vx, vy);

    // k2
    const x2 = x + 0.5 * dt * vx;
    const y2 = y + 0.5 * dt * vy;
    const vx2 = vx + 0.5 * dt * ax1;
    const vy2 = vy + 0.5 * dt * ay1;
    const [ax2, ay2] = acceleration1PN(x2, y2, vx2, vy2);

    // k3
    const x3 = x + 0.5 * dt * vx2;
    const y3 = y + 0.5 * dt * vy2;
    const vx3 = vx + 0.5 * dt * ax2;
    const vy3 = vy + 0.5 * dt * ay2;
    const [ax3, ay3] = acceleration1PN(x3, y3, vx3, vy3);

    // k4
    const x4 = x + dt * vx3;
    const y4 = y + dt * vy3;
    const vx4 = vx + dt * ax3;
    const vy4 = vy + dt * ay3;
    const [ax4, ay4] = acceleration1PN(x4, y4, vx4, vy4);

    // combine
    body.x += dt * (vx + 2 * vx2 + 2 * vx3 + vx4) / 6;
    body.y += dt * (vy + 2 * vy2 + 2 * vy3 + vy4) / 6;
    body.vx += dt * (ax1 + 2 * ax2 + 2 * ax3 + ax4) / 6;
    body.vy += dt * (ay1 + 2 * ay2 + 2 * ay3 + ay4) / 6;
  }

  // Planet config (semimajor axis in AU, eccentricity, color)
  const PLANET_CONFIGS = [
    { name: 'Mercury', a_AU: 0.38709893, e: 0.205630, color: '#c0c0c0', drawRadius: 3 },
    { name: 'Venus',   a_AU: 0.72333199, e: 0.006770, color: '#f7b267', drawRadius: 4 },
    { name: 'Earth',   a_AU: 1.00000011, e: 0.016710, color: '#4da3ff', drawRadius: 4 },
    { name: 'Mars',    a_AU: 1.52366231, e: 0.093400, color: '#ff6b6b', drawRadius: 4 }
  ];

  // Create initial state at perihelion (r along +x, v along +y)
  function createPlanetStates() {
    return PLANET_CONFIGS.map(cfg => {
      const a = cfg.a_AU * AU;
      const e = cfg.e;
      const rPeri = a * (1 - e);
      const vPeri = Math.sqrt(MU * (1 + e) / (a * (1 - e)));
      return {
        name: cfg.name,
        color: cfg.color,
        drawRadius: cfg.drawRadius,
        x: rPeri,
        y: 0,
        vx: 0,
        vy: vPeri,
        trail: [],
      };
    });
  }

  let planets = createPlanetStates();
  let simTime = 0; // seconds

  function resetSimulation() {
    planets = createPlanetStates();
    simTime = 0;
  }

  // UI wiring
  playPauseBtn.addEventListener('click', () => {
    running = !running;
    playPauseBtn.textContent = running ? 'Pause' : 'Play';
  });
  resetBtn.addEventListener('click', () => resetSimulation());
  relativityToggle.addEventListener('change', () => {
    modeLabel.textContent = `Mode: ${relativityToggle.checked ? '1PN' : 'Newtonian'}`;
  });
  speedSlider.addEventListener('input', () => {
    stepsPerFrame = Number(speedSlider.value);
    speedValue.textContent = `${stepsPerFrame}x`;
  });
  trailSlider.addEventListener('input', () => {
    trailMax = Number(trailSlider.value);
    trailValue.textContent = `${trailMax}`;
    for (const p of planets) {
      if (p.trail.length > trailMax) p.trail = p.trail.slice(-trailMax);
    }
  });

  // World -> screen mapping
  function getViewScaleAU() {
    const sel = scaleSelect.value;
    if (sel === 'auto') return 'auto';
    return Number(sel);
  }

  function computeAutoFitHalfSpanAU() {
    // Compute max distance among planets; include aphelion safety margin
    let maxR = 0;
    for (const p of planets) {
      const r = norm(p.x, p.y) / AU;
      if (r > maxR) maxR = r;
    }
    // Add margin so orbits fit comfortably
    return Math.max(2, Math.min(5, maxR * 1.2));
  }

  function worldToScreen(x, y) {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    const cx = w / 2, cy = h / 2;
    let halfSpanAU = getViewScaleAU();
    if (halfSpanAU === 'auto') halfSpanAU = computeAutoFitHalfSpanAU();
    const metersPerAU = AU;
    const metersPerPixel = (halfSpanAU * metersPerAU) / Math.min(w, h);
    const px = cx + (x / metersPerPixel);
    const py = cy - (y / metersPerPixel);
    return [px, py, metersPerPixel];
  }

  // Draw helpers
  function drawBackground() {
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function drawSun() {
    const rect = canvas.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    ctx.save();
    ctx.shadowColor = '#ffdf6e';
    ctx.shadowBlur = 20;
    ctx.fillStyle = '#ffcc33';
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function updateTrails() {
    for (const p of planets) {
      p.trail.push([p.x, p.y]);
      if (p.trail.length > trailMax) p.trail.shift();
    }
  }

  function drawTrails() {
    ctx.lineWidth = 1.25;
    for (const p of planets) {
      ctx.strokeStyle = p.color + 'cc';
      ctx.beginPath();
      let first = true;
      for (const [x, y] of p.trail) {
        const [sx, sy] = worldToScreen(x, y);
        if (first) {
          ctx.moveTo(sx, sy); first = false;
        } else {
          ctx.lineTo(sx, sy);
        }
      }
      ctx.stroke();
    }
  }

  function drawPlanets() {
    for (const p of planets) {
      const [sx, sy, metersPerPixel] = worldToScreen(p.x, p.y);
      // Visual draw radius in pixels; scale slightly with distance for readability
      const pxRadius = p.drawRadius;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(sx, sy, pxRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawHUD() {
    timeLabel.textContent = `t = ${(simTime / YEARS).toFixed(2)} years`;
  }

  // Main animation loop
  function stepSimulation() {
    if (running) {
      for (let i = 0; i < stepsPerFrame; i++) {
        for (const p of planets) rk4Step(p, BASE_DT);
        simTime += BASE_DT;
        updateTrails();
      }
    }

    drawBackground();
    drawSun();
    drawTrails();
    drawPlanets();
    drawHUD();

    requestAnimationFrame(stepSimulation);
  }

  // Kick off
  stepSimulation();
})();
