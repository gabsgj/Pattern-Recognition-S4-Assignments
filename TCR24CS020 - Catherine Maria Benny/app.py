import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# =========================
# HMM Baum-Welch Class
# =========================
class HMMBaumWelch:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations
        self.A = self._random_stochastic((self.N, self.N))
        self.B = self._random_stochastic((self.N, self.M))
        self.pi = self._random_stochastic((self.N,))
        
    def _random_stochastic(self, shape):
        x = np.random.rand(*shape)
        return x / x.sum(axis=-1, keepdims=True)

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, O[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]
        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N)
        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                )
        return beta

    def compute_gamma_xi(self, O, alpha, beta):
        T = len(O)
        gamma = np.zeros((T, self.N))
        xi = np.zeros((T-1, self.N, self.N))
        P_O = np.sum(alpha[-1])

        for t in range(T):
            gamma[t] = (alpha[t] * beta[t]) / P_O

        for t in range(T-1):
            denom = np.sum(
                alpha[t][:, None] *
                self.A *
                self.B[:, O[t+1]] *
                beta[t+1]
            )
            for i in range(self.N):
                numer = (
                    alpha[t, i] *
                    self.A[i] *
                    self.B[:, O[t+1]] *
                    beta[t+1]
                )
                xi[t, i] = numer / denom

        return gamma, xi, P_O

    def train(self, O, n_iter):
        history = []
        for _ in range(n_iter):
            alpha = self.forward(O)
            beta = self.backward(O)
            gamma, xi, P_O = self.compute_gamma_xi(O, alpha, beta)

            history.append({
                "A": self.A.copy(),
                "B": self.B.copy(),
                "pi": self.pi.copy(),
                "loglik": np.log(P_O)
            })

            self.pi = gamma[0]

            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

            for i in range(self.N):
                for k in range(self.M):
                    mask = (O == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

        return history


# =========================
# Streamlit UI
# =========================

st.title("Baum-Welch HMM Interactive Demo")

# Inputs
n_states = st.number_input("Number of Hidden States", min_value=2, max_value=6, value=2)
obs_sequence = st.text_input("Observation Sequence (comma separated integers)", "0,1,1,0,1")

O = np.array([int(x.strip()) for x in obs_sequence.split(",")])
n_obs = len(set(O))

iterations = st.slider("Number of Training Iterations", 1, 50, 10)

# Train model
hmm = HMMBaumWelch(n_states, n_obs)
history = hmm.train(O, iterations)

# Iteration viewer
iter_view = st.slider("View Iteration", 0, iterations-1, iterations-1)

current = history[iter_view]

st.subheader("Log-Likelihood Convergence")
fig1, ax1 = plt.subplots()
ax1.plot([h["loglik"] for h in history])
ax1.set_xlabel("Iteration")
ax1.set_ylabel("log P(O | λ)")
st.pyplot(fig1)

st.subheader("Transition Matrix A")
st.dataframe(pd.DataFrame(current["A"]))

st.subheader("Emission Matrix B")
st.dataframe(pd.DataFrame(current["B"]))

st.subheader("Initial Distribution π")
st.write(current["pi"])

# =========================
# Transition Graph
# =========================
st.subheader("State Transition Diagram")

G = nx.MultiDiGraph()

for i in range(n_states):
    G.add_node(f"S{i}")

for i in range(n_states):
    for j in range(n_states):
        prob = round(current["A"][i, j], 3)
        G.add_edge(f"S{i}", f"S{j}", label=str(prob))

pos = nx.circular_layout(G)
fig2, ax2 = plt.subplots()

nx.draw(G, pos, ax=ax2, with_labels=True, node_size=2500)

edge_labels = {(u, v, k): d["label"] 
               for u, v, k, d in G.edges(keys=True, data=True)}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2)

st.pyplot(fig2)