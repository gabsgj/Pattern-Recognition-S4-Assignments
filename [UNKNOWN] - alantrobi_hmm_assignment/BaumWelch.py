import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import networkx as nx
import tempfile
import os


# ============================================================
#                  FORMATTING HELPERS
# ============================================================

def pretty_matrix(mat, name):
    s = f"{name}:\n"
    for row in mat:
        s += "  " + "  ".join(f"{v:0.4f}" for v in row) + "\n"
    return s + "\n"


# ============================================================
#                  HMM / BAUM–WELCH LOGIC
# ============================================================

def forward(O, A, B, pi):
    N = A.shape[0]
    T = len(O)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, O[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, O[t]]

    return alpha


def backward(O, A, B):
    N = A.shape[0]
    T = len(O)
    beta = np.ones((T, N))

    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, O[t + 1]] * beta[t + 1])

    return beta


def compute_gamma_xi(O, A, B, alpha, beta):
    T, N = alpha.shape
    gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)

    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        denom = np.sum(alpha[t][:, None] * A * B[:, O[t + 1]] * beta[t + 1])
        for i in range(N):
            xi[t, i] = alpha[t, i] * A[i] * B[:, O[t + 1]] * beta[t + 1] / denom

    return gamma, xi


def one_iteration(O, A, B, pi):
    alpha = forward(O, A, B, pi)
    beta = backward(O, A, B)
    gamma, xi = compute_gamma_xi(O, A, B, alpha, beta)

    log_prob = np.log(np.sum(alpha[-1]))

    pi_new = gamma[0]

    A_new = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

    B_new = np.zeros_like(B)
    for i in range(B.shape[0]):
        for k in range(B.shape[1]):
            B_new[i, k] = np.sum(gamma[O == k, i])
        B_new[i] /= np.sum(gamma[:, i])

    return A_new, B_new, pi_new, log_prob


# ============================================================
#        BUILD TRANSITION DIAGRAM (NO GRAPHVIZ REQUIRED)
# ============================================================

def build_transition_image(A):
    temp_dir = os.path.join(tempfile.gettempdir(), "hmm_diagrams")
    os.makedirs(temp_dir, exist_ok=True)

    image_path = os.path.join(temp_dir, "transition.png")

    N = A.shape[0]
    G = nx.DiGraph()
    edge_labels = {}

    for i in range(N):
        for j in range(N):
            if A[i, j] > 0.001:
                G.add_edge(f"S{i}", f"S{j}")
                edge_labels[(f"S{i}", f"S{j}")] = f"{A[i,j]:.2f}"

    plt.figure(figsize=(5, 5))
    pos = nx.circular_layout(G)

    nx.draw(
        G, pos,
        with_labels=True,
        arrows=True,
        node_size=2000,
        node_color="#66b3ff",
        font_size=12,
        font_weight="bold",
        linewidths=1.5,
        edge_color="black"
    )

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return image_path


# ============================================================
#                        TKINTER GUI
# ============================================================

root = tk.Tk()
root.title("HMM Trainer — Single Window")
root.geometry("1100x800")

# ------------------ Top Controls ------------------

ctrl_frame = ttk.Frame(root, padding=10)
ctrl_frame.pack(fill="x")

ttk.Label(ctrl_frame, text="Observation Sequence (comma-separated):").grid(row=0, column=0, sticky="w")
entry_obs = ttk.Entry(ctrl_frame, width=30)
entry_obs.grid(row=0, column=1, padx=5)

ttk.Label(ctrl_frame, text="Hidden States (N):").grid(row=1, column=0, sticky="w")
entry_states = ttk.Entry(ctrl_frame, width=10)
entry_states.grid(row=1, column=1, sticky="w")

ttk.Label(ctrl_frame, text="Symbols (M):").grid(row=2, column=0, sticky="w")
entry_symbols = ttk.Entry(ctrl_frame, width=10)
entry_symbols.grid(row=2, column=1, sticky="w")

ttk.Label(ctrl_frame, text="Iterations:").grid(row=3, column=0, sticky="w")
entry_iters = ttk.Entry(ctrl_frame, width=10)
entry_iters.grid(row=3, column=1, sticky="w")


# ------------------ Main Frames ------------------

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_frame = ttk.Frame(main_frame)
left_frame.pack(side="left", fill="both", expand=True)

right_frame = ttk.Frame(main_frame)
right_frame.pack(side="right", fill="both", expand=True)


# ------------------ Left: Live Convergence Graph ------------------

fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Live Log-Likelihood")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log-Likelihood")
ax.grid(True)

canvas_graph = FigureCanvasTkAgg(fig, master=left_frame)
canvas_graph.get_tk_widget().pack(fill="both", expand=True)


# ------------------ Right: Transition Diagram ------------------

fig2 = Figure(figsize=(5, 4), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.axis("off")

canvas_diagram = FigureCanvasTkAgg(fig2, master=right_frame)
canvas_diagram.get_tk_widget().pack(fill="both", expand=True)


# ------------------ Bottom Output Box ------------------

output_box = scrolledtext.ScrolledText(root, width=120, height=12)
output_box.pack(pady=5)


# ============================================================
#                 ANIMATION / TRAINING LOOP
# ============================================================

animation_running = False
A = B = pi = None
O = None
iter_target = 0
iteration_counter = 0
log_values = []


def start_training():
    global A, B, pi, O, iter_target, iteration_counter, animation_running, log_values

    try:
        O_str = entry_obs.get().replace(" ", "")
        O = np.array([int(x) for x in O_str.split(",")])

        N = int(entry_states.get())
        M = int(entry_symbols.get())
        iter_target = int(entry_iters.get())

        np.random.seed(0)
        A = np.random.rand(N, N); A /= A.sum(axis=1, keepdims=True)
        B = np.random.rand(N, M); B /= B.sum(axis=1, keepdims=True)
        pi = np.random.rand(N);   pi /= pi.sum()

        iteration_counter = 0
        log_values = []

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "===== TRAINING STARTED =====\n\n")
        output_box.insert(tk.END, f"Total iterations: {iter_target}\n\n")

        ax.clear()
        ax.grid(True)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title("Live Log-Likelihood")

        ax2.clear()
        ax2.axis("off")

        animation_running = True
        animate()

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input:\n{e}")


def animate():
    global A, B, pi, iteration_counter, animation_running, log_values

    if not animation_running:
        return

    if iteration_counter >= iter_target:
        animation_running = False

        output_box.insert(tk.END, "\n===== TRAINING COMPLETE =====\n\n")
        output_box.insert(tk.END, pretty_matrix(A, "Final Transition Matrix A"))
        output_box.insert(tk.END, pretty_matrix(B, "Final Emission Matrix B"))

        output_box.insert(tk.END, "Initial Distribution pi:\n")
        output_box.insert(tk.END, "  " + "  ".join(f"{v:0.4f}" for v in pi) + "\n\n")
        output_box.insert(tk.END, f"Final Log-Likelihood: {log_values[-1]:.6f}\n\n")

        image_path = build_transition_image(A)
        if image_path:
            img = mpimg.imread(image_path)
            ax2.clear()
            ax2.imshow(img)
            ax2.axis("off")
            canvas_diagram.draw()

        return

    A, B, pi, loglik = one_iteration(O, A, B, pi)
    log_values.append(loglik)

    ax.clear()
    ax.grid(True)
    ax.set_title("Live Log-Likelihood")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.plot(log_values, marker="o")
    canvas_graph.draw()

    iteration_counter += 1
    root.after(100, animate)


ttk.Button(ctrl_frame, text="Start Training", command=start_training).grid(row=4, column=0, pady=10)

root.mainloop()