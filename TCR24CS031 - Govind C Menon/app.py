"""
HMM Baum-Welch — Flask Web Application
Pattern Recognition Assignment - CSE S4
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


# ─────────────────────────────────────────────
#  HMM CORE ALGORITHMS
# ─────────────────────────────────────────────

def forward_algorithm(obs, A, B, pi):
    T, N = len(obs), A.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]
    return alpha

def backward_algorithm(obs, A, B):
    T, N = len(obs), A.shape[0]
    beta = np.zeros((T, N))
    beta[T-1] = 1.0
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = np.sum(A[i, :] * B[:, obs[t+1]] * beta[t+1])
    return beta

def compute_gamma(alpha, beta):
    gamma = alpha * beta
    s = gamma.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-300
    return gamma / s

def compute_xi(obs, A, B, alpha, beta):
    T, N = len(obs), A.shape[0]
    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        denom = sum(alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j]
                    for i in range(N) for j in range(N)) or 1e-300
        for i in range(N):
            for j in range(N):
                xi[t][i][j] = alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j] / denom
    return xi

def baum_welch(obs_sequence, n_hidden, n_obs_symbols, max_iter=100, tol=1e-6, seed=42):
    np.random.seed(seed)
    obs = np.array(obs_sequence)
    T, N, M = len(obs), n_hidden, n_obs_symbols

    A  = np.random.dirichlet(np.ones(N), size=N)
    B  = np.random.dirichlet(np.ones(M), size=N)
    pi = np.random.dirichlet(np.ones(N))

    log_likelihoods, history = [], []

    for iteration in range(max_iter):
        alpha = forward_algorithm(obs, A, B, pi)
        beta  = backward_algorithm(obs, A, B)
        gamma = compute_gamma(alpha, beta)
        xi    = compute_xi(obs, A, B, alpha, beta)

        p_obs = max(np.sum(alpha[-1]), 1e-300)
        ll    = np.log(p_obs)
        log_likelihoods.append(ll)

        history.append({
            'iteration': iteration + 1,
            'alpha': alpha.tolist(),
            'beta':  beta.tolist(),
            'gamma': gamma.tolist(),
            'P_O_lambda': float(p_obs),
            'log_likelihood': float(ll),
        })

        # M-step
        pi_new = gamma[0]
        A_new  = np.zeros((N, N))
        for i in range(N):
            d = np.sum(gamma[:-1, i]) or 1e-300
            for j in range(N):
                A_new[i][j] = np.sum(xi[:, i, j]) / d
        B_new = np.zeros((N, M))
        for j in range(N):
            d = np.sum(gamma[:, j]) or 1e-300
            for k in range(M):
                B_new[j][k] = np.sum(gamma[obs == k, j]) / d

        pi_new /= (pi_new.sum() + 1e-300)
        A_new  /= (A_new.sum(axis=1, keepdims=True) + 1e-300)
        B_new  /= (B_new.sum(axis=1, keepdims=True) + 1e-300)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            A, B, pi = A_new, B_new, pi_new
            break
        A, B, pi = A_new, B_new, pi_new

    return A, B, pi, log_likelihoods, history


# ─────────────────────────────────────────────
#  VISUALIZATION → base64 PNG
# ─────────────────────────────────────────────

def make_plots(log_likelihoods, A, B, pi, obs, history):
    N, M = A.shape[0], B.shape[1]
    bg, text_c, acc, acc2 = '#0f0f1a', '#e0e0ff', '#7b61ff', '#00e5ff'
    grid_c = '#2a2a3a'
    node_colors = [acc, acc2, '#ff6b9d', '#ffd166']

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(bg)
    fig.suptitle('HMM Baum-Welch — Results', fontsize=18,
                 color=text_c, fontweight='bold', y=0.98)

    def style(ax):
        ax.set_facecolor('#12122a')
        for sp in ax.spines.values(): sp.set_color(grid_c)
        ax.tick_params(colors=text_c, labelsize=8)
        ax.xaxis.label.set_color(text_c)
        ax.yaxis.label.set_color(text_c)
        ax.title.set_color(text_c)

    iters = range(1, len(log_likelihoods)+1)

    # 1. log P(O|λ)
    ax1 = fig.add_subplot(3,3,1)
    ax1.plot(iters, log_likelihoods, color=acc, lw=2, marker='o', ms=3)
    ax1.fill_between(iters, log_likelihoods, alpha=0.15, color=acc)
    ax1.set_title('log P(O|λ) over Iterations'); ax1.set_xlabel('Iteration'); ax1.set_ylabel('log P(O|λ)')
    ax1.grid(True, color=grid_c, lw=0.5); style(ax1)

    # 2. Convergence delta
    ax2 = fig.add_subplot(3,3,2)
    if len(log_likelihoods) > 1:
        deltas = [abs(log_likelihoods[i]-log_likelihoods[i-1]) for i in range(1, len(log_likelihoods))]
        ax2.semilogy(range(2, len(log_likelihoods)+1), deltas, color=acc2, lw=2, marker='s', ms=3)
    ax2.set_title('Convergence |Δ log P(O|λ)|'); ax2.set_xlabel('Iteration'); ax2.set_ylabel('|Δ|')
    ax2.grid(True, color=grid_c, lw=0.5); style(ax2)

    # 3. Transition matrix A
    ax3 = fig.add_subplot(3,3,3)
    im = ax3.imshow(A, cmap='plasma', vmin=0, vmax=1)
    ax3.set_title('Transition Matrix A'); ax3.set_xlabel('To'); ax3.set_ylabel('From')
    ax3.set_xticks(range(N)); ax3.set_yticks(range(N))
    ax3.set_xticklabels([f'S{i}' for i in range(N)]); ax3.set_yticklabels([f'S{i}' for i in range(N)])
    for i in range(N):
        for j in range(N):
            ax3.text(j, i, f'{A[i,j]:.3f}', ha='center', va='center', color='white', fontsize=7, fontweight='bold')
    plt.colorbar(im, ax=ax3, fraction=0.046); style(ax3)

    # 4. Emission matrix B
    ax4 = fig.add_subplot(3,3,4)
    im2 = ax4.imshow(B, cmap='viridis', vmin=0, vmax=1)
    ax4.set_title('Emission Matrix B'); ax4.set_xlabel('Observation'); ax4.set_ylabel('State')
    ax4.set_xticks(range(M)); ax4.set_yticks(range(N))
    ax4.set_xticklabels([f'O{k}' for k in range(M)]); ax4.set_yticklabels([f'S{i}' for i in range(N)])
    for i in range(N):
        for j in range(M):
            ax4.text(j, i, f'{B[i,j]:.3f}', ha='center', va='center', color='white', fontsize=7, fontweight='bold')
    plt.colorbar(im2, ax=ax4, fraction=0.046); style(ax4)

    # 5. Initial distribution
    ax5 = fig.add_subplot(3,3,5)
    bars = ax5.bar([f'S{i}' for i in range(N)], pi, color=node_colors[:N], edgecolor='#ffffff30', lw=0.5)
    ax5.set_title('Initial Distribution π'); ax5.set_ylabel('Probability'); ax5.set_ylim(0,1)
    ax5.grid(True, axis='y', color=grid_c, lw=0.5)
    for bar, val in zip(bars, pi):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
                 ha='center', va='bottom', color=text_c, fontsize=8)
    style(ax5)

    # 6. Gamma
    ax6 = fig.add_subplot(3,3,6)
    gamma_last = np.array(history[-1]['gamma'])
    for i in range(N):
        ax6.plot(gamma_last[:, i], label=f'S{i}', lw=2, marker='o', ms=3)
    ax6.set_title('γ State Posteriors (Final)'); ax6.set_xlabel('Time t'); ax6.set_ylabel('P(q_t=i|O,λ)')
    ax6.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=text_c)
    ax6.grid(True, color=grid_c, lw=0.5); style(ax6)

    # 7. Alpha
    ax7 = fig.add_subplot(3,3,7)
    alpha_last = np.array(history[-1]['alpha'])
    for i in range(N):
        ax7.plot(alpha_last[:, i], label=f'S{i}', lw=2, marker='^', ms=3)
    ax7.set_title('α Forward Probs (Final)'); ax7.set_xlabel('Time t'); ax7.set_ylabel('α_t(i)')
    ax7.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=text_c)
    ax7.grid(True, color=grid_c, lw=0.5); style(ax7)

    # 8. Beta
    ax8 = fig.add_subplot(3,3,8)
    beta_last = np.array(history[-1]['beta'])
    for i in range(N):
        ax8.plot(beta_last[:, i], label=f'S{i}', lw=2, marker='v', ms=3)
    ax8.set_title('β Backward Probs (Final)'); ax8.set_xlabel('Time t'); ax8.set_ylabel('β_t(i)')
    ax8.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=text_c)
    ax8.grid(True, color=grid_c, lw=0.5); style(ax8)

    # 9. State transition diagram
    ax9 = fig.add_subplot(3,3,9)
    ax9.set_facecolor('#12122a'); ax9.set_xlim(-0.2,1.2); ax9.set_ylim(-0.2,1.2)
    ax9.set_aspect('equal'); ax9.axis('off')
    ax9.set_title('State Transition Diagram', color=text_c, fontsize=10)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    cx = 0.5 + 0.35*np.cos(angles); cy = 0.5 + 0.35*np.sin(angles)
    for i in range(N):
        for j in range(N):
            p = A[i][j]
            if p < 0.01: continue
            if i == j:
                circle = plt.Circle((cx[i]+0.08, cy[i]+0.08), 0.05, fill=False,
                                     color=node_colors[i%len(node_colors)], lw=1+3*p, ls='--')
                ax9.add_patch(circle)
                ax9.text(cx[i]+0.13, cy[i]+0.13, f'{p:.2f}', ha='center', va='center', color=text_c, fontsize=6)
            else:
                dx, dy = cx[j]-cx[i], cy[j]-cy[i]
                dist = np.sqrt(dx**2+dy**2) or 1e-9
                off = 0.06
                ax9.annotate("", xy=(cx[j]-off*dx/dist, cy[j]-off*dy/dist),
                              xytext=(cx[i]+off*dx/dist, cy[i]+off*dy/dist),
                              arrowprops=dict(arrowstyle='->', color=text_c, lw=1+3*p, alpha=0.7))
                mx = (cx[i]+cx[j])/2 + 0.03*(-dy/dist)
                my = (cy[i]+cy[j])/2 + 0.03*(dx/dist)
                ax9.text(mx, my, f'{p:.2f}', ha='center', va='center', color=text_c, fontsize=6,
                         bbox=dict(boxstyle='round,pad=0.1', fc='#12122a', ec='none'))
    for i in range(N):
        ax9.add_patch(plt.Circle((cx[i], cy[i]), 0.08, color=node_colors[i%len(node_colors)], zorder=5))
        ax9.text(cx[i], cy[i], f'S{i}', ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=6)

    plt.tight_layout(rect=[0,0,1,0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
#  HTML TEMPLATE
# ─────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HMM Baum-Welch</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080810;
    --surface: #0f0f1e;
    --surface2: #16162a;
    --border: #2a2a45;
    --accent: #7b61ff;
    --accent2: #00e5ff;
    --accent3: #ff6b9d;
    --text: #e0e0ff;
    --muted: #7070a0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
  }

  /* animated grid background */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(123,97,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(123,97,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }

  .container { position: relative; z-index: 1; max-width: 860px; margin: 0 auto; padding: 40px 20px 80px; }

  header { text-align: center; margin-bottom: 50px; }
  header .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent2);
    border: 1px solid var(--accent2);
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 2px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }
  header h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 12px;
  }
  header p { color: var(--muted); font-size: 15px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
  }
  .card h2 {
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card h2::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
  }

  .field { margin-bottom: 20px; }
  label {
    display: block;
    font-size: 12px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  label span { color: var(--accent3); margin-left: 4px; }

  input[type=text], input[type=number] {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 12px 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
    outline: none;
  }
  input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(123,97,255,0.15);
  }
  input::placeholder { color: var(--muted); }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

  .hint {
    font-size: 11px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 5px;
  }

  .checkbox-row {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
  }
  .checkbox-row input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); cursor: pointer; }
  .checkbox-row label { margin: 0; cursor: pointer; color: var(--text); font-size: 13px; text-transform: none; letter-spacing: 0; }

  .btn {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, var(--accent), #5a45cc);
    border: none;
    border-radius: 10px;
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    text-transform: uppercase;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(123,97,255,0.4); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

  /* Loading */
  #loading {
    display: none;
    text-align: center;
    padding: 40px;
    color: var(--accent2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 2px;
  }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Error */
  #error {
    display: none;
    background: rgba(255,107,157,0.1);
    border: 1px solid var(--accent3);
    border-radius: 10px;
    padding: 16px 20px;
    color: var(--accent3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    margin-bottom: 24px;
  }

  /* Results */
  #results { display: none; }

  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .metric {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }
  .metric .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 700;
    color: var(--accent2);
    margin-bottom: 4px;
  }
  .metric .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    margin-bottom: 20px;
  }
  th {
    background: var(--surface2);
    color: var(--accent);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface2); }

  .section-title {
    font-size: 12px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }

  .viz-img {
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
  }

  .iter-details {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.8;
    max-height: 320px;
    overflow-y: auto;
    color: var(--muted);
  }
  .iter-details .iter-head { color: var(--accent2); font-weight: 700; margin-top: 12px; }
  .iter-details .iter-head:first-child { margin-top: 0; }
  .iter-details .kv span { color: var(--text); }

  .download-btn {
    display: inline-block;
    padding: 10px 24px;
    background: var(--surface2);
    border: 1px solid var(--accent);
    border-radius: 8px;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    text-decoration: none;
    cursor: pointer;
    transition: background 0.2s;
    margin-top: 12px;
  }
  .download-btn:hover { background: rgba(123,97,255,0.15); }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">Pattern Recognition · CSE S4</div>
    <h1>HMM Baum-Welch<br>Algorithm</h1>
    <p>Enter your observed sequence and hidden state count to train the model</p>
  </header>

  <div id="error"></div>

  <div class="card">
    <h2>Input Parameters</h2>

    <div class="field">
      <label>Observed Sequence <span>*</span></label>
      <input type="text" id="obs" placeholder="e.g. 0 0 1 0 1 1 0 0 0 1" value="0 0 1 0 1 1 0 0 0 1 1 0 1 0 0">
      <div class="hint">Space-separated integers (0-indexed). Example: 0 1 0 2 1 0</div>
    </div>

    <div class="grid2">
      <div class="field">
        <label>Hidden States <span>*</span></label>
        <input type="number" id="n_hidden" value="2" min="1" max="10" placeholder="e.g. 2">
        <div class="hint">Number of hidden states (N)</div>
      </div>
      <div class="field">
        <label>Observation Symbols</label>
        <input type="number" id="n_obs" value="" min="1" max="20" placeholder="Auto-detect">
        <div class="hint">Leave blank to auto-detect from sequence</div>
      </div>
    </div>

    <div class="grid2">
      <div class="field">
        <label>Max Iterations</label>
        <input type="number" id="max_iter" value="100" min="1" max="1000" placeholder="100">
      </div>
      <div class="field">
        <label>Tolerance</label>
        <input type="text" id="tol" value="1e-6" placeholder="1e-6">
      </div>
    </div>

    <div class="field">
      <div class="checkbox-row">
        <input type="checkbox" id="show_details">
        <label for="show_details">Show per-iteration α / β / γ details (first 3 iterations)</label>
      </div>
    </div>
  </div>

  <button class="btn" id="runBtn" onclick="runHMM()">▶ Run Baum-Welch</button>

  <div id="loading">
    <div class="spinner"></div>
    COMPUTING...
  </div>

  <div id="results">
    <div class="card" style="margin-top:32px">
      <h2>Results</h2>
      <div class="metrics-grid" id="metrics"></div>

      <div class="section-title">Transition Matrix A</div>
      <div id="tableA"></div>

      <div class="section-title">Emission Matrix B</div>
      <div id="tableB"></div>

      <div class="section-title">Initial Distribution π</div>
      <div id="tablePi"></div>
    </div>

    <div class="card" id="detailsCard" style="display:none">
      <h2>Per-Iteration Details (α, β, γ)</h2>
      <div class="iter-details" id="iterDetails"></div>
    </div>

    <div class="card">
      <h2>Visualizations</h2>
      <img id="vizImg" class="viz-img" alt="HMM Visualization">
      <br>
      <a class="download-btn" id="downloadBtn" download="hmm_visualization.png">⬇ Download Image</a>
    </div>
  </div>
</div>

<script>
async function runHMM() {
  const btn = document.getElementById('runBtn');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const errDiv = document.getElementById('error');

  errDiv.style.display = 'none';
  results.style.display = 'none';

  const obsRaw = document.getElementById('obs').value.trim();
  if (!obsRaw) { showError('Please enter an observed sequence.'); return; }

  const obs = obsRaw.trim().split(/[ \t]+/).map(Number);
  if (obs.some(isNaN) || obs.some(v => v < 0)) {
    showError('Observed sequence must be non-negative integers (0-indexed).'); return;
  }

  const n_hidden = parseInt(document.getElementById('n_hidden').value);
  if (!n_hidden || n_hidden < 1) { showError('Hidden states must be at least 1.'); return; }

  const n_obs_raw = document.getElementById('n_obs').value.trim();
  const n_obs = n_obs_raw ? parseInt(n_obs_raw) : null;
  const max_iter = parseInt(document.getElementById('max_iter').value) || 100;
  const tol = parseFloat(document.getElementById('tol').value) || 1e-6;
  const show_details = document.getElementById('show_details').checked;

  btn.disabled = true;
  loading.style.display = 'block';

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ obs, n_hidden, n_obs, max_iter, tol, show_details })
    });
    const data = await resp.json();

    if (data.error) { showError(data.error); return; }

    renderResults(data);
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth' });

  } catch(e) {
    showError('Server error: ' + e.message);
  } finally {
    btn.disabled = false;
    loading.style.display = 'none';
  }
}

function showError(msg) {
  const e = document.getElementById('error');
  e.textContent = '⚠ ' + msg;
  e.style.display = 'block';
  document.getElementById('loading').style.display = 'none';
  document.getElementById('runBtn').disabled = false;
}

function renderResults(data) {
  const { A, B, pi, log_likelihoods, history, converged_at, final_p_obs, image_b64 } = data;
  const N = A.length, M = B[0].length;

  // Metrics
  document.getElementById('metrics').innerHTML = `
    <div class="metric"><div class="val">${N}</div><div class="lbl">Hidden States</div></div>
    <div class="metric"><div class="val">${M}</div><div class="lbl">Obs Symbols</div></div>
    <div class="metric"><div class="val">${converged_at}</div><div class="lbl">Iterations</div></div>
    <div class="metric"><div class="val">${final_p_obs}</div><div class="lbl">P(O|λ)</div></div>
    <div class="metric"><div class="val">${log_likelihoods[log_likelihoods.length-1].toFixed(4)}</div><div class="lbl">log P(O|λ)</div></div>
  `;

  // Table A
  let hA = '<table><tr><th></th>' + A.map((_,j)=>`<th>→ S${j}</th>`).join('') + '</tr>';
  A.forEach((row,i) => {
    hA += `<tr><td><b>S${i}</b></td>` + row.map(v=>`<td>${v.toFixed(5)}</td>`).join('') + '</tr>';
  });
  document.getElementById('tableA').innerHTML = hA + '</table>';

  // Table B
  let hB = '<table><tr><th></th>' + B[0].map((_,k)=>`<th>O${k}</th>`).join('') + '</tr>';
  B.forEach((row,i) => {
    hB += `<tr><td><b>S${i}</b></td>` + row.map(v=>`<td>${v.toFixed(5)}</td>`).join('') + '</tr>';
  });
  document.getElementById('tableB').innerHTML = hB + '</table>';

  // Table pi
  let hPi = '<table><tr>' + pi.map((_,i)=>`<th>S${i}</th>`).join('') + '</tr><tr>';
  hPi += pi.map(v=>`<td>${v.toFixed(5)}</td>`).join('') + '</tr></table>';
  document.getElementById('tablePi').innerHTML = hPi;

  // Per-iteration details
  const detailsCard = document.getElementById('detailsCard');
  const detailsDiv = document.getElementById('iterDetails');
  if (data.show_details && history && history.length > 0) {
    detailsCard.style.display = 'block';
    let html = '';
    history.slice(0, 3).forEach(h => {
      html += `<div class="iter-head">── Iteration ${h.iteration} ──</div>`;
      html += `<div class="kv">P(O|λ) = <span>${h.P_O_lambda.toExponential(6)}</span></div>`;
      html += `<div class="kv">log P(O|λ) = <span>${h.log_likelihood.toFixed(6)}</span></div>`;
      html += `<div class="kv">α(t=0) = <span>[${h.alpha[0].map(v=>v.toFixed(6)).join(', ')}]</span></div>`;
      html += `<div class="kv">β(t=0) = <span>[${h.beta[0].map(v=>v.toFixed(6)).join(', ')}]</span></div>`;
      html += `<div class="kv">γ(t=0) = <span>[${h.gamma[0].map(v=>v.toFixed(6)).join(', ')}]</span></div>`;
    });
    detailsDiv.innerHTML = html;
  } else {
    detailsCard.style.display = 'none';
  }

  // Image
  const img = document.getElementById('vizImg');
  img.src = 'data:image/png;base64,' + image_b64;
  document.getElementById('downloadBtn').href = 'data:image/png;base64,' + image_b64;
}
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/run', methods=['POST'])
def run():
    try:
        data = request.json
        obs          = data['obs']
        n_hidden     = int(data['n_hidden'])
        n_obs        = int(data['n_obs']) if data.get('n_obs') else max(obs) + 1
        max_iter     = int(data.get('max_iter', 100))
        tol          = float(data.get('tol', 1e-6))
        show_details = bool(data.get('show_details', False))

        if len(obs) < 2:
            return jsonify({'error': 'Need at least 2 observations.'})
        if n_hidden < 1:
            return jsonify({'error': 'Hidden states must be >= 1.'})
        if n_obs < max(obs) + 1:
            return jsonify({'error': f'Obs symbols must be >= {max(obs)+1}.'})

        A, B, pi, log_likelihoods, history = baum_welch(
            obs, n_hidden, n_obs, max_iter, tol, seed=42
        )

        image_b64 = make_plots(log_likelihoods, A, B, pi, obs, history)

        return jsonify({
            'A':              A.tolist(),
            'B':              B.tolist(),
            'pi':             pi.tolist(),
            'log_likelihoods': log_likelihoods,
            'history':        history if show_details else [],
            'show_details':   show_details,
            'converged_at':   len(log_likelihoods),
            'final_p_obs':    f"{np.exp(log_likelihoods[-1]):.4e}",
            'image_b64':      image_b64,
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  HMM Baum-Welch Web App")
    print("  Open in browser → http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
