/*
Relativistic Planetary Orbits (1PN) - Canvas simulation
- Units: AU, year, AU/year. Gravitational constant G and Sun mass combined as mu = G*M_sun in AU^3 / yr^2.
- Integrator: RK4 with adjustable time scaling.
- Gravity: Newtonian + 1PN correction term (simple approximation for Schwarzschild metric in harmonic coordinates for test particles).
  This captures perihelion precession qualitatively; do not use for precise ephemerides.
*/

(() => {
  // Physical constants in units: AU, year, AU/year
  const SPEED_OF_LIGHT_AU_PER_YEAR = 63241.077; // ~ 1c ≈ 63241 AU/year
  const MU_SUN = 4 * Math.PI * Math.PI; // G*M_sun ≈ 4π^2 AU^3/yr^2 (Kepler's 3rd law)

  // Rendering configuration
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const WIDTH = canvas.width; // logical pixels
  const HEIGHT = canvas.height;
  const CENTER_X = WIDTH / 2;
  const CENTER_Y = HEIGHT / 2;

  // Scale: AU to pixels (auto fit ~ Mars aphelion ~ 1.67 AU with margin)
  let AU_TO_PX = (Math.min(WIDTH, HEIGHT) * 0.42) / 1.8; // tweakable

  // UI elements
  const playPauseBtn = document.getElementById('playPause');
  const resetBtn = document.getElementById('reset');
  const speedSlider = document.getElementById('speed');
  const speedVal = document.getElementById('speedVal');
  const grSlider = document.getElementById('grFactor');
  const grVal = document.getElementById('grVal');
  const trailsChk = document.getElementById('trails');

  // Bodies: Sun fixed at origin (test-particle approximation)
  // Initial conditions from circular approximations at perihelion distances with known orbital speeds (AU/year)
  // More realistic ICs can be from NASA JPL, but we aim for stable visualization
  const bodies = [
    { name: 'Sun', color: '#ffcc66', radiusPx: 8, r: [0, 0], v: [0, 0], trail: [], trailColor: '#ffcc66' },
    // Mercury
    { name: 'Mercury', color: '#c1b9b1', radiusPx: 3, r: [0.3075, 0], v: [0, 10.086], trail: [], trailColor: '#a8a29e' },
    // Venus
    { name: 'Venus', color: '#ffd59e', radiusPx: 5, r: [0.7184, 0], v: [0, 7.382], trail: [], trailColor: '#fbbf24' },
    // Earth
    { name: 'Earth', color: '#8ecae6', radiusPx: 5, r: [1.0, 0], v: [0, 6.283185307179586], trail: [], trailColor: '#60a5fa' }, // ~2π AU/yr ≈ 1 AU/year*2π
    // Mars
    { name: 'Mars', color: '#f87171', radiusPx: 4, r: [1.666, 0], v: [0, 5.088], trail: [], trailColor: '#ef4444' },
  ];

  // Legend overlay
  createLegend();

  // Simulation state
  let isRunning = false;
  let grFactor = parseFloat(grSlider.value);
  let timeScale = parseFloat(speedSlider.value); // multiplier for dt
  let dtBase = 1 / 365.25; // 1 day in years
  let dt = dtBase * timeScale;

  playPauseBtn.addEventListener('click', () => {
    isRunning = !isRunning;
    playPauseBtn.textContent = isRunning ? 'Pause' : 'Play';
  });

  resetBtn.addEventListener('click', () => {
    resetSimulation();
  });

  speedSlider.addEventListener('input', () => {
    timeScale = parseFloat(speedSlider.value);
    dt = dtBase * timeScale;
    speedVal.textContent = `${timeScale.toFixed(1)}x`;
  });

  grSlider.addEventListener('input', () => {
    grFactor = parseFloat(grSlider.value);
    grVal.textContent = grFactor.toFixed(2);
  });

  trailsChk.addEventListener('change', () => {
    // no-op, checked in draw loop
  });

  function resetSimulation() {
    // Reset ICs
    bodies[0].r = [0, 0]; bodies[0].v = [0, 0]; bodies[0].trail = [];
    bodies[1].r = [0.3075, 0]; bodies[1].v = [0, 10.086]; bodies[1].trail = [];
    bodies[2].r = [0.7184, 0]; bodies[2].v = [0, 7.382]; bodies[2].trail = [];
    bodies[3].r = [1.0, 0]; bodies[3].v = [0, 2*Math.PI]; bodies[3].trail = [];
    bodies[4].r = [1.666, 0]; bodies[4].v = [0, 5.088]; bodies[4].trail = [];
  }

  // Compute acceleration with Newtonian + 1PN correction for test particle around a central mass
  function computeAcceleration(r, v, mu, c, alpha) {
    // r: [x,y], v: [vx,vy]
    const x = r[0];
    const y = r[1];
    const vx = v[0];
    const vy = v[1];
    const r2 = x*x + y*y;
    const r1 = Math.sqrt(r2);
    const invR = 1 / r1;
    const invR2 = 1 / r2;
    const invR3 = invR2 * invR;

    // Newtonian acceleration
    const aN_x = -mu * x * invR3;
    const aN_y = -mu * y * invR3;

    if (alpha === 0) return [aN_x, aN_y];

    // 1PN correction (approx): a = a_N + (mu/(c^2 r^3)) * [ (4mu/r - v^2) r + 4 (r·v) v ]
    // This is a common test-particle 1PN form (Schwarzschild). Multiply by alpha to scale.
    const v2 = vx*vx + vy*vy;
    const rdotv = x*vx + y*vy;
    const factor = mu / (c*c) * invR3;

    const coeff = 4*mu*invR - v2; // scalar multiplies r vector
    const aPN_x = factor * (coeff * x + 4 * rdotv * vx);
    const aPN_y = factor * (coeff * y + 4 * rdotv * vy);

    return [aN_x + alpha * aPN_x, aN_y + alpha * aPN_y];
  }

  function rk4Step(state, dt, mu, c, alpha) {
    // state: { r: [x,y], v: [vx,vy] }
    const r0 = state.r;
    const v0 = state.v;

    // k1
    const a1 = computeAcceleration(r0, v0, mu, c, alpha);

    // k2
    const r1 = [ r0[0] + 0.5*dt * v0[0], r0[1] + 0.5*dt * v0[1] ];
    const v1 = [ v0[0] + 0.5*dt * a1[0], v0[1] + 0.5*dt * a1[1] ];
    const a2 = computeAcceleration(r1, v1, mu, c, alpha);

    // k3
    const r2 = [ r0[0] + 0.5*dt * v1[0], r0[1] + 0.5*dt * v1[1] ];
    const v2 = [ v0[0] + 0.5*dt * a2[0], v0[1] + 0.5*dt * a2[1] ];
    const a3 = computeAcceleration(r2, v2, mu, c, alpha);

    // k4
    const r3 = [ r0[0] + dt * v2[0], r0[1] + dt * v2[1] ];
    const v3 = [ v0[0] + dt * a3[0], v0[1] + dt * a3[1] ];
    const a4 = computeAcceleration(r3, v3, mu, c, alpha);

    // Combine
    const rx = r0[0] + (dt/6) * (v0[0] + 2*v1[0] + 2*v2[0] + v3[0]);
    const ry = r0[1] + (dt/6) * (v0[1] + 2*v1[1] + 2*v2[1] + v3[1]);

    const vx = v0[0] + (dt/6) * (a1[0] + 2*a2[0] + 2*a3[0] + a4[0]);
    const vy = v0[1] + (dt/6) * (a1[1] + 2*a2[1] + 2*a3[1] + a4[1]);

    return { r: [rx, ry], v: [vx, vy] };
  }

  function worldToScreen(x, y) {
    return [ CENTER_X + x * AU_TO_PX, CENTER_Y - y * AU_TO_PX ];
  }

  function draw() {
    // Background and subtle fade for trails
    if (!trailsChk.checked) {
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
    } else {
      ctx.fillStyle = 'rgba(8, 11, 18, 0.22)';
      ctx.fillRect(0, 0, WIDTH, HEIGHT);
    }

    // draw Sun glow
    {
      const [sx, sy] = worldToScreen(0, 0);
      const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, 40);
      grd.addColorStop(0, 'rgba(255, 210, 100, 0.95)');
      grd.addColorStop(1, 'rgba(255, 210, 100, 0.0)');
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(sx, sy, 40, 0, Math.PI*2);
      ctx.fill();
    }

    // Draw orbits/trails
    for (let i = 1; i < bodies.length; i++) {
      const b = bodies[i];
      if (b.trail.length > 1) {
        ctx.strokeStyle = b.trailColor;
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        for (let k = 0; k < b.trail.length; k++) {
          const [tx, ty] = worldToScreen(b.trail[k][0], b.trail[k][1]);
          if (k === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
        }
        ctx.stroke();
      }
    }

    // Draw bodies
    for (const b of bodies) {
      const [sx, sy] = worldToScreen(b.r[0], b.r[1]);
      ctx.fillStyle = b.color;
      ctx.beginPath();
      ctx.arc(sx, sy, b.radiusPx, 0, Math.PI * 2);
      ctx.fill();
    }

    // Crosshair at origin
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.moveTo(CENTER_X - 8, CENTER_Y);
    ctx.lineTo(CENTER_X + 8, CENTER_Y);
    ctx.moveTo(CENTER_X, CENTER_Y - 8);
    ctx.lineTo(CENTER_X, CENTER_Y + 8);
    ctx.stroke();
  }

  function step() {
    // Advance planets (Sun fixed)
    for (let i = 1; i < bodies.length; i++) {
      const b = bodies[i];
      const next = rk4Step({ r: b.r, v: b.v }, dt, MU_SUN, SPEED_OF_LIGHT_AU_PER_YEAR, grFactor);
      b.r = next.r;
      b.v = next.v;

      // Trails
      if (trailsChk.checked) {
        b.trail.push([b.r[0], b.r[1]]);
        if (b.trail.length > 2000) b.trail.shift();
      } else {
        b.trail.length = 0;
      }
    }
  }

  // Animation loop
  function loop() {
    if (isRunning) {
      // Integrate multiple substeps for smoother motion at high speeds
      const substeps = Math.max(1, Math.floor(timeScale));
      const subDtOld = dt;
      dt = dtBase * timeScale / substeps;
      for (let s = 0; s < substeps; s++) step();
      dt = subDtOld; // restore
    }
    draw();
    requestAnimationFrame(loop);
  }

  function createLegend() {
    const legend = document.createElement('div');
    legend.className = 'legend';
    const entries = [
      { name: 'Sun', color: '#ffcc66' },
      { name: 'Mercury', color: '#a8a29e' },
      { name: 'Venus', color: '#fbbf24' },
      { name: 'Earth', color: '#60a5fa' },
      { name: 'Mars', color: '#ef4444' },
    ];
    for (const e of entries) {
      const row = document.createElement('div');
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.background = e.color;
      const label = document.createElement('span');
      label.textContent = e.name;
      row.appendChild(dot);
      row.appendChild(label);
      legend.appendChild(row);
    }
    document.body.appendChild(legend);
  }

  // Kickoff
  resetSimulation();
  draw();
  requestAnimationFrame(loop);
})();
