/* ====================================================
   HMM Baum-Welch Website — script.js
   Full implementation: hero canvas, state machine,
   Baum-Welch EM algorithm, Viterbi decoding
   ==================================================== */

// ─── UTILITY ────────────────────────────────────────
const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Safe log
const safeLog = v => (v <= 0 ? -1e10 : Math.log(v));

// ─── NAVBAR TOGGLE ──────────────────────────────────
document.getElementById('navToggle').addEventListener('click', () => {
  document.querySelector('.nav-links').classList.toggle('open');
});

// Close nav on link click
document.querySelectorAll('.nav-links a').forEach(a => {
  a.addEventListener('click', () =>
    document.querySelector('.nav-links').classList.remove('open'));
});

// ─── HERO PARTICLE CANVAS ───────────────────────────
(function heroCanvas() {
  const canvas = $('heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  resize();
  window.addEventListener('resize', () => { resize(); initParticles(); });

  function initParticles() {
    particles = Array.from({ length: 80 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 2 + 0.5,
      alpha: Math.random() * 0.5 + 0.1,
    }));
  }

  initParticles();

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(244,63,94,${0.04 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(244,63,94,${p.alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }
  draw();
})();

// ─── LIVE STATE MACHINE CANVAS ──────────────────────
(function stateMachineAnim() {
  const canvas = $('stateMachineCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const smObs = $('smObs');

  // Example 3-state machine (S0, S1, S2)
  const states = [
    { x: W / 2, y: 55,       label: 'S₀', color: '#f43f5e' },
    { x: W / 2 - 80, y: 165, label: 'S₁', color: '#fb7185' },
    { x: W / 2 + 80, y: 165, label: 'S₂', color: '#fda4af' },
  ];

  const transitions = [
    { from: 0, to: 1, prob: 0.7 },
    { from: 1, to: 2, prob: 0.6 },
    { from: 2, to: 0, prob: 0.5 },
    { from: 0, to: 2, prob: 0.3 },
    { from: 2, to: 1, prob: 0.5 },
  ];

  const obsLabels = ['Low', 'Medium', 'High'];
  let currentState = 0;
  let nextState = 0;
  let progress = 0;
  let animating = false;
  let stepTimer = null;

  function getNext(from) {
    const out = transitions.filter(t => t.from === from);
    const r = Math.random();
    let cumul = 0;
    for (const t of out) {
      cumul += t.prob;
      if (r < cumul) return t.to;
    }
    return out.length ? out[out.length - 1].to : (from + 1) % states.length;
  }

  function drawArrow(x1, y1, x2, y2, color, opacity = 1) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    const ux = dx / len, uy = dy / len;
    const r = 22;
    const sx = x1 + ux * r, sy = y1 + uy * r;
    const ex = x2 - ux * r, ey = y2 - uy * r;

    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrowhead
    const angle = Math.atan2(ey - sy, ex - sx);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - 8 * Math.cos(angle - 0.4), ey - 8 * Math.sin(angle - 0.4));
    ctx.lineTo(ex - 8 * Math.cos(angle + 0.4), ey - 8 * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawState(s, active, pulse) {
    const r = 24;
    ctx.save();
    // Glow for active
    if (active) {
      ctx.shadowColor = s.color;
      ctx.shadowBlur = 20 + pulse * 10;
    }
    // Circle
    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.fillStyle = active
      ? s.color
      : 'rgba(19,25,41,0.9)';
    ctx.fill();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = active ? 2.5 : 1.5;
    ctx.globalAlpha = active ? 1 : 0.5;
    ctx.stroke();
    ctx.restore();

    // Label
    ctx.save();
    ctx.fillStyle = active ? '#fff' : s.color;
    ctx.globalAlpha = active ? 1 : 0.6;
    ctx.font = `bold 13px Syne, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(s.label, s.x, s.y);
    ctx.restore();
  }

  let pulseT = 0;

  function draw() {
    ctx.clearRect(0, 0, W, H);
    pulseT += 0.05;
    const pulse = (Math.sin(pulseT) + 1) / 2;

    // Draw all transitions
    transitions.forEach(t => {
      drawArrow(
        states[t.from].x, states[t.from].y,
        states[t.to].x, states[t.to].y,
        '#4a5568', 0.5
      );
    });

    // Draw states
    states.forEach((s, i) => drawState(s, i === currentState, pulse));

    // Draw traveling particle
    if (animating) {
      const from = states[currentState];
      const to = states[nextState];
      const px = from.x + (to.x - from.x) * progress;
      const py = from.y + (to.y - from.y) * progress;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.shadowColor = '#fff';
      ctx.shadowBlur = 12;
      ctx.fill();
    }

    requestAnimationFrame(draw);
  }

  draw();

  function step() {
    nextState = getNext(currentState);
    animating = true;
    progress = 0;
    const obsIdx = Math.floor(Math.random() * obsLabels.length);
    if (smObs) smObs.textContent = `Current Observation: O${obsIdx+1} — ${obsLabels[obsIdx]} Emission`;

    const anim = setInterval(() => {
      progress += 0.04;
      if (progress >= 1) {
        progress = 1;
        clearInterval(anim);
        animating = false;
        currentState = nextState;
      }
    }, 16);
  }

  stepTimer = setInterval(step, 1800);
})();

// ─── SPINNER BUTTONS ────────────────────────────────
document.querySelectorAll('.spin-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const targetId = btn.dataset.target;
    const dir = parseFloat(btn.dataset.dir);
    const inp = $(targetId);
    if (!inp) return;
    let v = parseFloat(inp.value) + dir;
    const mn = parseFloat(inp.min || 0);
    const mx = parseFloat(inp.max || 9999);
    inp.value = clamp(v, mn, mx);
  });
});

// ─── BAUM-WELCH CORE IMPLEMENTATION ─────────────────

/**
 * Initialize random row-stochastic matrix
 */
function randomStochastic(rows, cols) {
  return Array.from({ length: rows }, () => {
    const row = Array.from({ length: cols }, () => Math.random() + 0.1);
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map(v => v / sum);
  });
}

/**
 * Forward algorithm — returns alpha[t][i] and log-likelihood
 */
function forward(obs, pi, A, B) {
  const T = obs.length, N = pi.length;
  const alpha = Array.from({ length: T }, () => new Array(N).fill(0));
  const c = new Array(T).fill(0);

  // Init
  for (let i = 0; i < N; i++) alpha[0][i] = pi[i] * B[i][obs[0]];
  c[0] = alpha[0].reduce((a, b) => a + b, 0) || 1e-300;
  for (let i = 0; i < N; i++) alpha[0][i] /= c[0];

  // Recursion
  for (let t = 1; t < T; t++) {
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += alpha[t - 1][i] * A[i][j];
      alpha[t][j] = s * B[j][obs[t]];
    }
    c[t] = alpha[t].reduce((a, b) => a + b, 0) || 1e-300;
    for (let j = 0; j < N; j++) alpha[t][j] /= c[t];
  }

  const logLik = c.reduce((acc, ci) => acc + Math.log(ci), 0);
  return { alpha, c, logLik };
}

/**
 * Backward algorithm — returns beta[t][i] (scaled by same c)
 */
function backward(obs, A, B, c) {
  const T = obs.length, N = A.length;
  const beta = Array.from({ length: T }, () => new Array(N).fill(0));

  // Init
  for (let i = 0; i < N; i++) beta[T - 1][i] = 1 / (c[T - 1] || 1e-300);

  // Recursion
  for (let t = T - 2; t >= 0; t--) {
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j];
      beta[t][i] = s / (c[t] || 1e-300);
    }
  }

  return beta;
}

/**
 * Compute gamma[t][i] and xi[t][i][j]
 */
function computeGammaXi(obs, alpha, beta, A, B) {
  const T = obs.length, N = A.length;
  const gamma = Array.from({ length: T }, () => new Array(N).fill(0));
  const xi = Array.from({ length: T - 1 }, () =>
    Array.from({ length: N }, () => new Array(N).fill(0))
  );

  for (let t = 0; t < T; t++) {
    let denom = 0;
    for (let i = 0; i < N; i++) denom += alpha[t][i] * beta[t][i];
    denom = denom || 1e-300;
    for (let i = 0; i < N; i++) gamma[t][i] = (alpha[t][i] * beta[t][i]) / denom;
  }

  for (let t = 0; t < T - 1; t++) {
    let denom = 0;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        denom += alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j];
    denom = denom || 1e-300;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]) / denom;
  }

  return { gamma, xi };
}

/**
 * M-Step: update pi, A, B
 */
function mStep(obs, gamma, xi, N, M) {
  const T = obs.length;

  // New pi
  const piNew = gamma[0].slice();

  // New A
  const ANew = Array.from({ length: N }, (_, i) => {
    const denomA = gamma.slice(0, T - 1).reduce((s, g) => s + g[i], 0) || 1e-300;
    return Array.from({ length: N }, (_, j) => {
      const num = xi.reduce((s, x) => s + x[i][j], 0);
      return num / denomA;
    });
  });

  // New B
  const BNew = Array.from({ length: N }, (_, i) => {
    const denomB = gamma.reduce((s, g) => s + g[i], 0) || 1e-300;
    return Array.from({ length: M }, (_, k) => {
      const num = gamma.reduce((s, g, t) => s + (obs[t] === k ? g[i] : 0), 0);
      return num / denomB;
    });
  });

  return { piNew, ANew, BNew };
}

/**
 * Full Baum-Welch training
 */
function baumWelch(obs, N, M, maxIter = 50, tol = 1e-6) {
  let pi = Array(N).fill(1 / N);
  let A = randomStochastic(N, N);
  let B = randomStochastic(N, M);
  const logLikelihoods = [];

  for (let iter = 0; iter < maxIter; iter++) {
    const { alpha, c, logLik } = forward(obs, pi, A, B);
    const beta = backward(obs, A, B, c);
    const { gamma, xi } = computeGammaXi(obs, alpha, beta, A, B);
    const { piNew, ANew, BNew } = mStep(obs, gamma, xi, N, M);

    logLikelihoods.push(logLik);
    pi = piNew; A = ANew; B = BNew;

    if (iter > 0 && Math.abs(logLikelihoods[iter] - logLikelihoods[iter - 1]) < tol) {
      logLikelihoods.push(logLik); // push final
      break;
    }
  }

  return { pi, A, B, logLikelihoods };
}

// ─── VITERBI IMPLEMENTATION ──────────────────────────

function viterbi(obs, pi, A, B) {
  const T = obs.length, N = pi.length;
  const delta = Array.from({ length: T }, () => new Array(N).fill(0));
  const psi   = Array.from({ length: T }, () => new Array(N).fill(0));

  // Init
  for (let i = 0; i < N; i++) delta[0][i] = Math.log(pi[i] + 1e-300) + Math.log(B[i][obs[0]] + 1e-300);

  // Recursion
  for (let t = 1; t < T; t++) {
    for (let j = 0; j < N; j++) {
      let maxVal = -Infinity, maxState = 0;
      for (let i = 0; i < N; i++) {
        const v = delta[t - 1][i] + Math.log(A[i][j] + 1e-300);
        if (v > maxVal) { maxVal = v; maxState = i; }
      }
      delta[t][j] = maxVal + Math.log(B[j][obs[t]] + 1e-300);
      psi[t][j] = maxState;
    }
  }

  // Traceback
  const path = new Array(T);
  let best = 0;
  for (let i = 1; i < N; i++) if (delta[T - 1][i] > delta[T - 1][best]) best = i;
  path[T - 1] = best;
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];

  return { path, delta };
}

// ─── CHART DRAWING (no external lib) ────────────────

function drawConvergenceChart(canvasEl, logLikelihoods) {
  const ctx = canvasEl.getContext('2d');
  const W = canvasEl.offsetWidth;
  canvasEl.width = W;
  canvasEl.height = 220;
  const H = 220;
  const pad = { top: 24, right: 24, bottom: 40, left: 60 };
  const data = logLikelihoods;
  const n = data.length;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(0,0,0,0.2)';
  ctx.roundRect(0, 0, W, H, 8);
  ctx.fill();

  if (n < 2) return;

  const minY = Math.min(...data);
  const maxY = Math.max(...data);
  const rangeY = maxY - minY || 1;

  const xScale = (i) => pad.left + (i / (n - 1)) * (W - pad.left - pad.right);
  const yScale = (v) => H - pad.bottom - ((v - minY) / rangeY) * (H - pad.top - pad.bottom);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad.top + (g / 4) * (H - pad.top - pad.bottom);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();

    // Y label
    const val = maxY - (g / 4) * rangeY;
    ctx.fillStyle = '#4a5568';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
  }

  // X labels
  ctx.fillStyle = '#4a5568';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 8))) {
    ctx.fillText(i + 1, xScale(i), H - pad.bottom + 16);
  }

  // Filled area
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(data[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xScale(i), yScale(data[i]));
  ctx.lineTo(xScale(n - 1), H - pad.bottom);
  ctx.lineTo(xScale(0), H - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = 'rgba(244,63,94,0.06)';
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(data[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xScale(i), yScale(data[i]));
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Dots
  for (let i = 0; i < n; i++) {
    ctx.beginPath();
    ctx.arc(xScale(i), yScale(data[i]), 3, 0, Math.PI * 2);
    ctx.fillStyle = i === n - 1 ? '#f59e0b' : '#00e5ff';
    ctx.fill();
  }

  // Convergence line
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(245,158,11,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, yScale(data[n - 1]));
  ctx.lineTo(W - pad.right, yScale(data[n - 1]));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f59e0b';
  ctx.font = '10px Syne, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('Converged', W - pad.right, yScale(data[n - 1]) - 6);
}

// ─── MATRIX HTML ────────────────────────────────────

function matrixToHTML(matrix, rowLabels, colLabels, title) {
  let html = `<div style="margin-bottom:24px;"><h4 style="color:var(--accent1);font-size:14px;margin-bottom:10px;">${title}</h4>`;
  html += '<table class="matrix-table"><thead><tr><th></th>';
  colLabels.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  matrix.forEach((row, i) => {
    html += `<tr><th>${rowLabels[i]}</th>`;
    row.forEach(v => html += `<td>${typeof v === 'number' ? v.toFixed(4) : v}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

// ─── TRAIN BUTTON ────────────────────────────────────
$('trainBtn').addEventListener('click', () => {
  const N = clamp(parseInt($('nStates').value), 2, 5);
  const M = clamp(parseInt($('mSymbols').value), 2, 5);
  const maxIter = clamp(parseInt($('maxIter').value), 5, 200);
  const rawObs = $('obsInput').value;

  // Parse observation sequence
  let obs;
  try {
    obs = rawObs.split(',').map(s => parseInt(s.trim()));
    if (obs.some(isNaN)) throw new Error('NaN');
    if (obs.some(v => v < 0 || v >= M)) {
      alert(`Observations must be integers in range 0 to ${M - 1} for M=${M} symbols.`);
      return;
    }
  } catch {
    alert('Invalid observation sequence. Use comma-separated integers like: 0,1,0,1');
    return;
  }

  if (obs.length < 2) {
    alert('Please enter at least 2 observations.');
    return;
  }

  // Run Baum-Welch
  const { pi, A, B, logLikelihoods } = baumWelch(obs, N, M, maxIter);

  // Show output
  const output = $('playgroundOutput');
  output.style.display = 'block';
  output.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Draw convergence chart
  const chartCanvas = $('convergenceChart');
  setTimeout(() => drawConvergenceChart(chartCanvas, logLikelihoods), 100);

  // Result matrices
  const stateLabels = Array.from({ length: N }, (_, i) => `S${i}`);
  const symLabels = Array.from({ length: M }, (_, i) => `V${i}`);
  const matrices = $('resultMatrices');
  matrices.innerHTML =
    `<h4 style="font-size:18px;margin-bottom:20px;color:var(--text-primary);">Training Results — Iteration ${logLikelihoods.length} / ${maxIter}</h4>` +
    `<div style="margin-bottom:12px;padding:12px 16px;background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.15);border-radius:8px;font-family:var(--font-math);font-size:13px;color:var(--accent4);">
      Final log-likelihood: <strong>${logLikelihoods[logLikelihoods.length - 1].toFixed(6)}</strong> &nbsp;|&nbsp;
      Improvement: <strong>${(logLikelihoods[logLikelihoods.length - 1] - logLikelihoods[0]).toFixed(4)}</strong>
    </div>` +
    matrixToHTML(A, stateLabels, stateLabels, 'Transition Matrix (A)') +
    matrixToHTML(B, stateLabels, symLabels, 'Emission Matrix (B)') +
    matrixToHTML([pi], ['π'], stateLabels, 'Initial Distribution (π)');

  // Log
  const logArea = $('logArea');
  logArea.innerHTML = logLikelihoods.map((v, i) =>
    `Iter ${String(i + 1).padStart(3, '0')}  log P(O|λ) = ${v.toFixed(8)}`
  ).join('\n');

  // Store model for Viterbi
  window._trainedModel = { pi, A, B, N, M };
});

// ─── VITERBI DECODER ────────────────────────────────

$('decodeBtn').addEventListener('click', () => {
  const rawObs = $('viterbiObs').value;

  // Use trained model if available, else use example parameters
  let pi, A, B, N, M;

  if (window._trainedModel) {
    ({ pi, A, B, N, M } = window._trainedModel);
  } else {
    // Default: Rainy/Sunny example, binary obs (0=W, 1=H)
    N = 2; M = 2;
    pi = [0.6, 0.4];
    A = [[0.7, 0.3], [0.4, 0.6]];
    B = [[0.1, 0.9], [0.6, 0.4]];
  }

  let obs;
  try {
    obs = rawObs.split(',').map(s => parseInt(s.trim()));
    if (obs.some(isNaN)) throw new Error();
    if (obs.some(v => v < 0 || v >= M)) {
      alert(`Observations must be integers 0-${M - 1}. ${window._trainedModel ? 'Train the model first to set M.' : ''}`);
      return;
    }
  } catch {
    alert('Invalid sequence. Use comma-separated integers like: 0,1,0,0,1');
    return;
  }

  const { path, delta } = viterbi(obs, pi, A, B);
  const stateLabels = Array.from({ length: N }, (_, i) => `S${i}`);

  // Display sequence
  const resultDiv = $('viterbiResult');
  let seqHTML = '<div class="state-sequence">';
  path.forEach((s, t) => {
    if (t > 0) seqHTML += '<span class="state-arrow">→</span>';
    seqHTML += `<span class="state-badge">${stateLabels[s]}</span>`;
  });
  seqHTML += '</div>';
  seqHTML += `<p class="viterbi-info">Most probable hidden state sequence decoded via dynamic programming traceback. Log-probability: <strong>${delta[obs.length - 1][path[obs.length - 1]].toFixed(4)}</strong></p>`;
  resultDiv.innerHTML = seqHTML;

  // Draw Trellis
  drawTrellis(obs, path, delta, N, stateLabels);
});

function drawTrellis(obs, path, delta, N, stateLabels) {
  const container = $('viterbiTrellis');
  const T = obs.length;
  const W = Math.max(400, T * 90 + 80);
  const H = N * 70 + 60;

  container.innerHTML = `<canvas id="trellisCanvas" width="${W}" height="${H}" style="max-width:100%;border-radius:8px;background:rgba(0,0,0,0.2);"></canvas>`;
  const canvas = $('trellisCanvas');
  const ctx = canvas.getContext('2d');

  const px = (t) => 60 + t * 90;
  const py = (i) => 40 + i * 70;

  // Normalize delta for coloring
  const allDelta = delta.flat();
  const minD = Math.min(...allDelta.filter(isFinite));
  const maxD = Math.max(...allDelta.filter(isFinite));

  // Draw edges
  for (let t = 0; t < T - 1; t++) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const onPath = path[t] === i && path[t + 1] === j;
        ctx.beginPath();
        ctx.moveTo(px(t), py(i));
        ctx.lineTo(px(t + 1), py(j));
        ctx.strokeStyle = onPath ? '#f59e0b' : 'rgba(74,85,104,0.3)';
        ctx.lineWidth = onPath ? 2.5 : 1;
        ctx.stroke();
      }
    }
  }

  // Draw nodes
  for (let t = 0; t < T; t++) {
    // Obs label
    ctx.fillStyle = '#4a5568';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`O${t}=${obs[t]}`, px(t), 18);

    for (let i = 0; i < N; i++) {
      const onPath = path[t] === i;
      const norm = (delta[t][i] - minD) / (maxD - minD || 1);

      ctx.beginPath();
      ctx.arc(px(t), py(i), 16, 0, Math.PI * 2);
      ctx.fillStyle = onPath ? '#fb7185' : `rgba(244,63,94,${0.1 + norm * 0.4})`;
      ctx.fill();
      ctx.strokeStyle = onPath ? '#fb7185' : '#f43f5e';
      ctx.lineWidth = onPath ? 2.5 : 1;
      ctx.globalAlpha = onPath ? 1 : 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;

      ctx.fillStyle = onPath ? '#fff' : '#8892a4';
      ctx.font = `bold 11px Syne, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(stateLabels[i], px(t), py(i) + 4);
    }
  }
}

// ─── SMOOTH SCROLL ──────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

// ─── MathJax RE-RENDER on dynamic content ───────────
function rerenderMath() {
  if (window.MathJax) {
    MathJax.typesetPromise && MathJax.typesetPromise();
  }
}

$('trainBtn').addEventListener('click', () => setTimeout(rerenderMath, 300));