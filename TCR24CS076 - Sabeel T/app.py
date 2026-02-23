import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="HMM Trainer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.big-title {
    font-size:28px !important;
    font-weight:700;
    color:#1f77b4;
}
.card {
    padding:15px;
    border-radius:10px;
    background-color:#f8f9fa;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 Hidden Markov Model Trainer</div>', unsafe_allow_html=True)
st.write("Train a Hidden Markov Model using the Baum-Welch Algorithm")

# =====================================
# HMM Functions
# =====================================

def normalize_rows(mat):
    return mat / mat.sum(axis=1, keepdims=True)

def normalize_vec(v):
    return v / v.sum()

def random_init(N, M):
    A = normalize_rows(np.random.rand(N, N))
    B = normalize_rows(np.random.rand(N, M))
    pi = normalize_vec(np.random.rand(N))
    return A, B, pi

def forward(A, B, pi, obs):
    T = len(obs)
    N = A.shape[0]
    alpha = np.zeros((T, N))
    c = np.zeros(T)

    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    loglik = -np.sum(np.log(c))
    return alpha, c, loglik

def backward(A, B, obs, c):
    T = len(obs)
    N = A.shape[0]
    beta = np.zeros((T, N))
    beta[T-1] = 1

    for t in reversed(range(T-1)):
        beta[t] = (A * (B[:, obs[t+1]] * beta[t+1])[None, :]).sum(axis=1)
        beta[t] /= c[t+1]

    return beta

def baum_welch(obs, N, M, iterations):
    A, B, pi = random_init(N, M)
    loglik_history = []

    for _ in range(iterations):
        alpha, c, loglik = forward(A, B, pi, obs)
        beta = backward(A, B, obs, c)

        T = len(obs)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            numerator = alpha[t][:,None] * A * (B[:,obs[t+1]] * beta[t+1])[None,:]
            xi[t] = numerator / numerator.sum()

        pi = gamma[0]
        A = normalize_rows(xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:,None])

        B_new = np.zeros((N, M))
        for k in range(M):
            mask = (obs == k)
            B_new[:,k] = gamma[mask].sum(axis=0)

        B = normalize_rows(B_new / gamma.sum(axis=0)[:,None])
        loglik_history.append(loglik)

    return A, B, pi, loglik_history

def encode_obs(symbols):
    unique = sorted(set(symbols))
    mapping = {s:i for i,s in enumerate(unique)}
    encoded = np.array([mapping[s] for s in symbols])
    return encoded, mapping

# =====================================
# Sidebar Inputs
# =====================================

with st.sidebar:
    st.header("⚙ Model Configuration")
    obs_input = st.text_input("Observation Sequence", "W,H,H,W,H")
    states = st.number_input("Hidden States", min_value=2, value=2)
    iterations = st.slider("Training Iterations", 5, 100, 25)
    train_button = st.button("🚀 Train Model")

# =====================================
# Training Section
# =====================================

if train_button:

    obs_symbols = obs_input.replace(",", " ").split()
    obs, mapping = encode_obs(obs_symbols)

    N = states
    M = len(mapping)

    A, B, pi, loglik_history = baum_welch(obs, N, M, iterations)

    # ===== Metrics Row =====
    col1, col2, col3 = st.columns(3)
    col1.metric("Hidden States", N)
    col2.metric("Observations", len(obs))
    col3.metric("Final Log-Likelihood", f"{loglik_history[-1]:.4f}")

    st.divider()

    # ===== Tabs Layout =====
    tab1, tab2, tab3 = st.tabs(["📈 Convergence", "🔁 Transition Graph", "📋 Parameters"])

    # =========================
    # Tab 1: Convergence Plot
    # =========================
    with tab1:
        fig, ax = plt.subplots()
        ax.plot(loglik_history, marker='o')
        ax.set_title("Log-Likelihood Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log P(O | λ)")
        ax.grid(True)
        st.pyplot(fig)

    # =========================
    # Tab 2: Transition Graph
    # =========================
    with tab2:
        dot = Digraph()
        dot.attr(rankdir="LR")

        obs_names = list(mapping.keys())

        for i in range(N):
            emission_text = "\n".join(
                [f"{obs_names[j]}: {B[i,j]:.2f}" for j in range(len(obs_names))]
            )
            label = f"State {i}\nπ={pi[i]:.2f}\n{emission_text}"
            dot.node(f"S{i}", label)

        for i in range(N):
            for j in range(N):
                if A[i,j] > 0.01:
                    dot.edge(f"S{i}", f"S{j}",
                             label=f"{A[i,j]:.2f}")

        st.graphviz_chart(dot)

    # =========================
    # Tab 3: Matrices
    # =========================
    with tab3:
        st.subheader("Transition Matrix (A)")
        st.dataframe(A.round(4), use_container_width=True)

        st.subheader("Emission Matrix (B)")
        st.dataframe(B.round(4), use_container_width=True)

        st.subheader("Initial Distribution (π)")
        st.dataframe(pi.round(4))

    st.success("Model training completed successfully!")