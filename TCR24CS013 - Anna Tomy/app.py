import streamlit as st
import numpy as np
from graphviz import Digraph


# ---------- Utility Functions ----------

def normalize_rows(mat):
    mat = np.maximum(mat, 1e-12)
    return mat / mat.sum(axis=1, keepdims=True)


def random_stochastic(shape):
    mat = np.random.rand(*shape)
    return normalize_rows(mat)


# ---------- Baum-Welch Algorithm ----------

def baum_welch(obs, A, B, pi, max_iters=50, tol=1e-6):
    N = A.shape[0]
    T = len(obs)
    logs = []

    for iteration in range(max_iters):

        # ----- Forward -----
        alpha = np.zeros((T, N))
        alpha[0] = pi * B[:, obs[0]]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]

        # ----- Backward -----
        beta = np.zeros((T, N))
        beta[-1] = 1

        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])

        # ----- Gamma & Xi -----
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, N, N))

        for t in range(T - 1):
            denom = (alpha[t][:, None] * A * B[:, obs[t + 1]] * beta[t + 1]).sum()
            xi[t] = (
                alpha[t][:, None]
                * A
                * B[:, obs[t + 1]]
                * beta[t + 1]
            ) / denom

        # ----- Re-estimate -----

        pi_new = gamma[0]

        A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]

        M = B.shape[1]
        B_new = np.zeros_like(B)

        for k in range(M):
            mask = (obs == k)
            B_new[:, k] = gamma[mask].sum(axis=0)

        B_new /= gamma.sum(axis=0)[:, None]

        diff = np.max(np.abs(A_new - A))
        logs.append(f"Iteration {iteration+1} | change = {diff:.6f}")

        A, B, pi = A_new, B_new, pi_new

        if diff < tol:
            break

    return A, B, pi, logs


# ---------- State Diagram ----------

def draw_hmm(A):
    from graphviz import Digraph

def draw_hmm(A):
    N = A.shape[0]
    dot = Digraph(format="png")

    dot.attr(rankdir="LR")

    # Node style
    dot.attr(
        "node",
        shape="circle",
        style="filled",
        fillcolor="#6FA8DC",
        fontname="Helvetica",
        fontsize="14",
        penwidth="2"
    )

    for i in range(N):
        dot.node(f"S{i}", f"State {i}")

    # Edge style
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="12",
        color="#333333",
        penwidth="1.5"
    )

    # Add edges WITH labels
    for i in range(N):
        for j in range(N):
            prob = A[i, j]

            if prob > 0:   # show all transitions
                dot.edge(
                    f"S{i}",
                    f"S{j}",
                    label=f"{prob:.3f}"
                )

    return dot


# ---------- Streamlit UI ----------

st.title("Hidden Markov Model — Baum-Welch Training")
st.title("Hidden Markov Model — Baum-Welch Training")

st.markdown("""
This application demonstrates training of a Hidden Markov Model (HMM)
using the Baum-Welch algorithm (an Expectation–Maximization method).

Given an observation sequence, the model learns:

• State transition probabilities (A)  
• Emission probabilities (B)  
• Initial state probabilities (π)  

A visual state diagram is also generated to illustrate the learned model.
""")

st.divider()

st.write("Train an HMM using the Baum-Welch Algorithm")

# Inputs
N = st.number_input("Number of hidden states", 2, 10, 2)
M = st.number_input("Number of observation symbols", 2, 10, 2)

obs_input = st.text_input(
    "Observation sequence (space separated integers)",
    "0 1 1 0 1"
)

max_iters = st.number_input("Max iterations", 1, 500, 50)
tol = st.text_input("Tolerance", "1e-6")

advanced = st.checkbox("Advanced: Provide A, B, π manually")


# ---------- Run Button ----------

if st.button("Train Model"):

    obs = np.array(list(map(int, obs_input.split())))
    tol = float(tol)

    if advanced:

        st.info("Enter matrices as Python lists")

        A = np.array(eval(st.text_area("Transition matrix A")))
        B = np.array(eval(st.text_area("Emission matrix B")))
        pi = np.array(eval(st.text_area("Initial probabilities π")))

    else:

        A = random_stochastic((N, N))
        B = random_stochastic((N, M))
        pi = np.random.rand(N)
        pi /= pi.sum()

    A_new, B_new, pi_new, logs = baum_welch(
        obs, A, B, pi, max_iters, tol
    )

    st.success("Training complete!")

    st.subheader("Transition Matrix A")
    st.write(A_new)

    st.subheader("Emission Matrix B")
    st.write(B_new)

    st.subheader("Initial State Probabilities π")
    st.write(pi_new)

    st.subheader("State Diagram")
    st.graphviz_chart(draw_hmm(A_new))

    with st.expander("Iteration Log"):
        for log in logs:
            st.write(log)