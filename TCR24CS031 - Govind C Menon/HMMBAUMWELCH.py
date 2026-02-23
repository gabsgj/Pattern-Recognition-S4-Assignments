"""
HMM Baum-Welch Algorithm Implementation
Pattern Recognition Assignment - CSE S4
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix for Windows/Python 3.14 Tkinter crash
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')


def forward_algorithm(obs, A, B, pi):
    """
    Forward pass: compute alpha (forward probabilities)
    alpha[t][i] = P(O_1, O_2, ..., O_t, q_t = i | lambda)
    """
    T = len(obs)
    N = A.shape[0]
    alpha = np.zeros((T, N))

    # Initialization
    alpha[0] = pi * B[:, obs[0]]

    # Induction
    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]

    return alpha


def backward_algorithm(obs, A, B):
    """
    Backward pass: compute beta (backward probabilities)
    beta[t][i] = P(O_{t+1}, ..., O_T | q_t = i, lambda)
    """
    T = len(obs)
    N = A.shape[0]
    beta = np.zeros((T, N))

    # Initialization
    beta[T-1] = 1.0

    # Induction (going backwards)
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = np.sum(A[i, :] * B[:, obs[t+1]] * beta[t+1])

    return beta


def compute_gamma(alpha, beta):
    """
    gamma[t][i] = P(q_t = i | O, lambda)
    Probability of being in state i at time t given the full observation sequence.
    """
    gamma = alpha * beta
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma_sum[gamma_sum == 0] = 1e-300  # avoid division by zero
    gamma = gamma / gamma_sum
    return gamma


def compute_xi(obs, A, B, alpha, beta):
    """
    xi[t][i][j] = P(q_t = i, q_{t+1} = j | O, lambda)
    Probability of transitioning from state i to state j at time t.
    """
    T = len(obs)
    N = A.shape[0]
    xi = np.zeros((T-1, N, N))

    for t in range(T-1):
        denom = 0.0
        for i in range(N):
            for j in range(N):
                denom += alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j]

        if denom == 0:
            denom = 1e-300

        for i in range(N):
            for j in range(N):
                xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j]) / denom

    return xi


def baum_welch(obs_sequence, n_hidden_states, n_obs_symbols=None,
               max_iter=100, tol=1e-6, random_seed=42):
    """
    Baum-Welch Algorithm (EM for HMM)

    Parameters:
        obs_sequence   : list of observed state indices (0-indexed)
        n_hidden_states: number of hidden states (N)
        n_obs_symbols  : number of distinct observation symbols (M)
        max_iter       : maximum iterations
        tol            : convergence tolerance
        random_seed    : for reproducibility

    Returns:
        A    : transition matrix (N x N)
        B    : emission matrix (N x M)
        pi   : initial state distribution (N,)
        log_likelihoods : P(O|lambda) at each iteration
        history : dict with alpha, beta, gamma, xi per iteration
    """
    np.random.seed(random_seed)
    obs = np.array(obs_sequence)
    T = len(obs)
    N = n_hidden_states
    M = n_obs_symbols if n_obs_symbols is not None else max(obs) + 1

    # --- Random initialization ---
    # Transition matrix A
    A = np.random.dirichlet(np.ones(N), size=N)
    # Emission matrix B
    B = np.random.dirichlet(np.ones(M), size=N)
    # Initial distribution pi
    pi = np.random.dirichlet(np.ones(N))

    log_likelihoods = []
    history = []

    print("=" * 60)
    print("  HMM BAUM-WELCH ALGORITHM")
    print("=" * 60)
    print(f"  Observations  : {obs_sequence}")
    print(f"  Hidden States : {N}")
    print(f"  Obs Symbols   : {M}")
    print(f"  Sequence Len  : {T}")
    print("=" * 60)

    for iteration in range(max_iter):
        # ---- E-STEP ----
        alpha = forward_algorithm(obs, A, B, pi)
        beta  = backward_algorithm(obs, A, B)
        gamma = compute_gamma(alpha, beta)
        xi    = compute_xi(obs, A, B, alpha, beta)

        # P(O | lambda)
        p_obs = np.sum(alpha[-1])
        if p_obs <= 0:
            p_obs = 1e-300
        log_likelihood = np.log(p_obs)
        log_likelihoods.append(log_likelihood)

        # Store history (optional detailed output)
        history.append({
            'iteration': iteration + 1,
            'alpha': alpha.copy(),
            'beta': beta.copy(),
            'gamma': gamma.copy(),
            'xi': xi.copy(),
            'P_O_lambda': p_obs,
            'log_likelihood': log_likelihood,
            'A': A.copy(),
            'B': B.copy(),
            'pi': pi.copy()
        })

        # ---- M-STEP ----
        # Update pi
        pi_new = gamma[0]

        # Update A
        A_new = np.zeros((N, N))
        for i in range(N):
            denom = np.sum(gamma[:-1, i])
            if denom == 0:
                denom = 1e-300
            for j in range(N):
                A_new[i][j] = np.sum(xi[:, i, j]) / denom

        # Update B
        B_new = np.zeros((N, M))
        for j in range(N):
            denom = np.sum(gamma[:, j])
            if denom == 0:
                denom = 1e-300
            for k in range(M):
                mask = (obs == k)
                B_new[j][k] = np.sum(gamma[mask, j]) / denom

        # Normalize to ensure valid distributions
        pi_new = pi_new / (pi_new.sum() + 1e-300)
        A_new  = A_new / (A_new.sum(axis=1, keepdims=True) + 1e-300)
        B_new  = B_new / (B_new.sum(axis=1, keepdims=True) + 1e-300)

        # Check convergence
        if iteration > 0:
            delta = abs(log_likelihoods[-1] - log_likelihoods[-2])
            if delta < tol:
                print(f"\n  Converged at iteration {iteration + 1}  (Δ = {delta:.2e})")
                A, B, pi = A_new, B_new, pi_new
                break

        A, B, pi = A_new, B_new, pi_new

        if (iteration + 1) % 10 == 0 or iteration < 5:
            print(f"  Iter {iteration+1:4d} | log P(O|λ) = {log_likelihood:.6f}")

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    print("\n  Initial Distribution π:")
    for i, p in enumerate(pi):
        print(f"    State {i}: {p:.6f}")

    print("\n  Transition Matrix A:")
    header = "        " + "  ".join([f"  S{j}" for j in range(N)])
    print(header)
    for i in range(N):
        row = f"  S{i} -> " + "  ".join([f"{A[i][j]:.4f}" for j in range(N)])
        print(row)

    print("\n  Emission Matrix B:")
    header = "        " + "  ".join([f"  O{k}" for k in range(M)])
    print(header)
    for i in range(N):
        row = f"  S{i}    " + "  ".join([f"{B[i][k]:.4f}" for k in range(M)])
        print(row)

    final_p = np.exp(log_likelihoods[-1])
    print(f"\n  Final P(O|λ) = {final_p:.8e}")
    print(f"  log P(O|λ)  = {log_likelihoods[-1]:.6f}")
    print("=" * 60)

    return A, B, pi, log_likelihoods, history


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(log_likelihoods, A, B, pi, obs_sequence, history):
    """Generate all required visualizations."""

    N = A.shape[0]
    M = B.shape[1]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f0f1a')

    # Shared style
    text_color = '#e0e0ff'
    accent     = '#7b61ff'
    accent2    = '#00e5ff'
    grid_color = '#2a2a3a'

    plt.rcParams.update({
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
    })

    fig.suptitle('HMM Baum-Welch Algorithm — Visualization',
                 fontsize=18, color=text_color, fontweight='bold', y=0.98)

    # ── 1. log P(O|λ) over iterations ──────────────────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_facecolor('#12122a')
    iters = range(1, len(log_likelihoods) + 1)
    ax1.plot(iters, log_likelihoods, color=accent, linewidth=2, marker='o', markersize=3)
    ax1.fill_between(iters, log_likelihoods, alpha=0.15, color=accent)
    ax1.set_title('log P(O|λ) over Iterations', color=text_color, fontsize=10)
    ax1.set_xlabel('Iteration', fontsize=8)
    ax1.set_ylabel('log P(O|λ)', fontsize=8)
    ax1.grid(True, color=grid_color, linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)

    # ── 2. 1 - P(O|λ_new) / P(O|λ_old)  change ────────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_facecolor('#12122a')
    if len(log_likelihoods) > 1:
        deltas = [abs(log_likelihoods[i] - log_likelihoods[i-1])
                  for i in range(1, len(log_likelihoods))]
        ax2.semilogy(range(2, len(log_likelihoods) + 1), deltas,
                     color=accent2, linewidth=2, marker='s', markersize=3)
        ax2.set_title('Convergence: |Δ log P(O|λ)|', color=text_color, fontsize=10)
        ax2.set_xlabel('Iteration', fontsize=8)
        ax2.set_ylabel('|Δ log-likelihood|', fontsize=8)
        ax2.grid(True, color=grid_color, linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color(grid_color)

    # ── 3. Transition Matrix Heatmap ───────────────────────────────
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.set_facecolor('#12122a')
    im = ax3.imshow(A, cmap='plasma', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Transition Matrix A', color=text_color, fontsize=10)
    ax3.set_xlabel('To State', fontsize=8)
    ax3.set_ylabel('From State', fontsize=8)
    ax3.set_xticks(range(N))
    ax3.set_yticks(range(N))
    ax3.set_xticklabels([f'S{i}' for i in range(N)], fontsize=8)
    ax3.set_yticklabels([f'S{i}' for i in range(N)], fontsize=8)
    for i in range(N):
        for j in range(N):
            ax3.text(j, i, f'{A[i,j]:.3f}', ha='center', va='center',
                     color='white', fontsize=7, fontweight='bold')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    for spine in ax3.spines.values():
        spine.set_color(grid_color)

    # ── 4. Emission Matrix Heatmap ─────────────────────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_facecolor('#12122a')
    im2 = ax4.imshow(B, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax4.set_title('Emission Matrix B', color=text_color, fontsize=10)
    ax4.set_xlabel('Observation Symbol', fontsize=8)
    ax4.set_ylabel('Hidden State', fontsize=8)
    ax4.set_xticks(range(M))
    ax4.set_yticks(range(N))
    ax4.set_xticklabels([f'O{k}' for k in range(M)], fontsize=8)
    ax4.set_yticklabels([f'S{i}' for i in range(N)], fontsize=8)
    for i in range(N):
        for j in range(M):
            ax4.text(j, i, f'{B[i,j]:.3f}', ha='center', va='center',
                     color='white', fontsize=7, fontweight='bold')
    plt.colorbar(im2, ax=ax4, fraction=0.046)
    for spine in ax4.spines.values():
        spine.set_color(grid_color)

    # ── 5. Initial Distribution ────────────────────────────────────
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_facecolor('#12122a')
    colors_pi = [accent, accent2, '#ff6b9d', '#ffd166'][:N]
    bars = ax5.bar([f'S{i}' for i in range(N)], pi,
                   color=colors_pi[:N], edgecolor='#ffffff30', linewidth=0.5)
    ax5.set_title('Initial Distribution π', color=text_color, fontsize=10)
    ax5.set_ylabel('Probability', fontsize=8)
    ax5.set_ylim(0, 1)
    ax5.grid(True, axis='y', color=grid_color, linewidth=0.5)
    for bar, val in zip(bars, pi):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', color=text_color, fontsize=8)
    for spine in ax5.spines.values():
        spine.set_color(grid_color)

    # ── 6. Gamma (state posteriors) at last iteration ──────────────
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.set_facecolor('#12122a')
    gamma_last = history[-1]['gamma']
    T = gamma_last.shape[0]
    for i in range(N):
        ax6.plot(range(T), gamma_last[:, i],
                 label=f'State {i}', linewidth=2, marker='o', markersize=3)
    ax6.set_title('γ (State Posteriors) — Final Iter', color=text_color, fontsize=10)
    ax6.set_xlabel('Time t', fontsize=8)
    ax6.set_ylabel('P(q_t = i | O, λ)', fontsize=8)
    ax6.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=text_color)
    ax6.grid(True, color=grid_color, linewidth=0.5)
    for spine in ax6.spines.values():
        spine.set_color(grid_color)

    # ── 7. Alpha (forward) at last iteration ───────────────────────
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_facecolor('#12122a')
    alpha_last = history[-1]['alpha']
    for i in range(N):
        ax7.plot(range(T), alpha_last[:, i],
                 label=f'State {i}', linewidth=2, marker='^', markersize=3)
    ax7.set_title('α (Forward Probs) — Final Iter', color=text_color, fontsize=10)
    ax7.set_xlabel('Time t', fontsize=8)
    ax7.set_ylabel('α_t(i)', fontsize=8)
    ax7.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=text_color)
    ax7.grid(True, color=grid_color, linewidth=0.5)
    for spine in ax6.spines.values():
        spine.set_color(grid_color)

    # ── 8. Beta (backward) at last iteration ───────────────────────
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.set_facecolor('#12122a')
    beta_last = history[-1]['beta']
    for i in range(N):
        ax8.plot(range(T), beta_last[:, i],
                 label=f'State {i}', linewidth=2, marker='v', markersize=3)
    ax8.set_title('β (Backward Probs) — Final Iter', color=text_color, fontsize=10)
    ax8.set_xlabel('Time t', fontsize=8)
    ax8.set_ylabel('β_t(i)', fontsize=8)
    ax8.legend(fontsize=7, facecolor='#1a1a2e', labelcolor=text_color)
    ax8.grid(True, color=grid_color, linewidth=0.5)
    for spine in ax8.spines.values():
        spine.set_color(grid_color)

    # ── 9. State Transition Diagram (drawn manually) ───────────────
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_facecolor('#12122a')
    ax9.set_xlim(-0.2, 1.2)
    ax9.set_ylim(-0.2, 1.2)
    ax9.set_aspect('equal')
    ax9.axis('off')
    ax9.set_title('State Transition Diagram', color=text_color, fontsize=10)

    # Place nodes in a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cx = 0.5 + 0.35 * np.cos(angles)
    cy = 0.5 + 0.35 * np.sin(angles)

    node_colors = [accent, accent2, '#ff6b9d', '#ffd166']

    # Draw edges
    for i in range(N):
        for j in range(N):
            prob = A[i][j]
            if prob < 0.01:
                continue
            lw = 1 + 3 * prob
            if i == j:
                # Self-loop
                circle = plt.Circle((cx[i] + 0.08, cy[i] + 0.08), 0.05,
                                     fill=False, color=node_colors[i % len(node_colors)],
                                     linewidth=lw, linestyle='--')
                ax9.add_patch(circle)
                ax9.text(cx[i] + 0.13, cy[i] + 0.13, f'{prob:.2f}',
                         ha='center', va='center', color=text_color, fontsize=6)
            else:
                dx = cx[j] - cx[i]
                dy = cy[j] - cy[i]
                dist = np.sqrt(dx**2 + dy**2)
                offset = 0.06
                x1 = cx[i] + offset * dx / dist
                y1 = cy[i] + offset * dy / dist
                x2 = cx[j] - offset * dx / dist
                y2 = cy[j] - offset * dy / dist
                ax9.annotate("", xy=(x2, y2), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', color=text_color,
                                              lw=lw, alpha=0.7))
                mx = (x1 + x2) / 2 + 0.03 * (-dy / dist)
                my = (y1 + y2) / 2 + 0.03 * (dx / dist)
                ax9.text(mx, my, f'{prob:.2f}',
                         ha='center', va='center', color=text_color, fontsize=6,
                         bbox=dict(boxstyle='round,pad=0.1', fc='#12122a', ec='none'))

    # Draw nodes
    for i in range(N):
        circle = plt.Circle((cx[i], cy[i]), 0.08,
                             color=node_colors[i % len(node_colors)], zorder=5)
        ax9.add_patch(circle)
        ax9.text(cx[i], cy[i], f'S{i}',
                 ha='center', va='center', color='white',
                 fontsize=9, fontweight='bold', zorder=6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('hmm_visualization.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\n  Visualization saved → hmm_visualization.png")
    plt.close()
    print("  Open 'hmm_visualization.png' in your file explorer to view it.")


# ─────────────────────────────────────────────
#  MAIN — Example Usage
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  HMM BAUM-WELCH — Interactive Input")
    print("=" * 60)

    # ── Observed sequence ──────────────────────────────────────────
    print("\n  Enter the observed sequence as space-separated integers.")
    print("  Example: 0 0 1 0 1 1 0 0 1")
    while True:
        try:
            raw = input("  Observed sequence: ").strip()
            obs_sequence = list(map(int, raw.split()))
            if len(obs_sequence) < 2:
                print("  ⚠  Please enter at least 2 observations.")
                continue
            if min(obs_sequence) < 0:
                print("  ⚠  Observations must be non-negative integers (0-indexed).")
                continue
            break
        except ValueError:
            print("  ⚠  Invalid input. Enter integers separated by spaces.")

    # ── Number of hidden states ────────────────────────────────────
    while True:
        try:
            n_hidden_states = int(input("  Number of hidden states (e.g. 2): ").strip())
            if n_hidden_states < 1:
                print("  ⚠  Must be at least 1.")
                continue
            break
        except ValueError:
            print("  ⚠  Please enter a valid integer.")

    # ── Number of observation symbols (optional) ───────────────────
    auto_m = max(obs_sequence) + 1
    print(f"\n  Number of distinct observation symbols.")
    print(f"  (Press Enter to auto-detect: {auto_m})")
    while True:
        try:
            raw_m = input("  Number of observation symbols: ").strip()
            if raw_m == "":
                n_obs_symbols = auto_m
                break
            n_obs_symbols = int(raw_m)
            if n_obs_symbols < auto_m:
                print(f"  ⚠  Must be at least {auto_m} (highest symbol in sequence + 1).")
                continue
            break
        except ValueError:
            print("  ⚠  Please enter a valid integer.")

    # ── Optional: max iterations & tolerance ──────────────────────
    print(f"\n  Max iterations (Press Enter for default 100):")
    raw_iter = input("  Max iterations: ").strip()
    max_iter = int(raw_iter) if raw_iter.isdigit() else 100

    print(f"  Convergence tolerance (Press Enter for default 1e-6):")
    raw_tol = input("  Tolerance: ").strip()
    try:
        tol = float(raw_tol) if raw_tol else 1e-6
    except ValueError:
        tol = 1e-6

    # ── Show optional per-iteration details? ──────────────────────
    show_details = input("\n  Show per-iteration α/β/γ details? (y/N): ").strip().lower() == 'y'

    print()

    # ── Run Baum-Welch ─────────────────────────────────────────────
    A, B, pi, log_likelihoods, history = baum_welch(
        obs_sequence=obs_sequence,
        n_hidden_states=n_hidden_states,
        n_obs_symbols=n_obs_symbols,
        max_iter=max_iter,
        tol=tol,
        random_seed=42
    )

    # ── Optional per-iteration output ─────────────────────────────
    if show_details:
        print("\n  ── OPTIONAL: Per-Iteration Details (first 3 iterations) ──")
        for h in history[:3]:
            print(f"\n  --- Iteration {h['iteration']} ---")
            print(f"  P(O|λ)       = {h['P_O_lambda']:.6e}")
            print(f"  log P(O|λ)   = {h['log_likelihood']:.6f}")
            print(f"  Alpha (t=0): {h['alpha'][0]}")
            print(f"  Beta  (t=0): {h['beta'][0]}")
            print(f"  Gamma (t=0): {h['gamma'][0]}")

    # ── Generate visualizations ────────────────────────────────────
    plot_results(log_likelihoods, A, B, pi, obs_sequence, history)
