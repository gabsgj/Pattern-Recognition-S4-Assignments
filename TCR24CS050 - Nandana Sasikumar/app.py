import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the HMM Logic
class HMM_Streamlit:
    def __init__(self, states, symbols):
        self.states = states
        self.symbols = symbols
        self.N = len(states)
        self.M = len(symbols)
        # Random initializations
        self.pi = np.random.dirichlet(np.ones(self.N))
        self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)

    def run_iteration(self, obs_seq):
        T = len(obs_seq)
        # Forward Pass
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, obs_seq[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * self.B[j, obs_seq[t]]
        
        # Backward Pass
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N)
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1, :])
        
        prob_O = np.sum(alpha[-1]) 
        gamma = (alpha * beta) / prob_O
        
        xi = np.zeros((T-1, self.N, self.N))
        for t in range(T-1):
            for i in range(self.N):
                xi[t, i, :] = (alpha[t, i] * self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1, :]) / prob_O

        # Parameter Updates
        self.pi = gamma[0] 
        self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0).reshape(-1, 1) 
        for l in range(self.M):
            self.B[:, l] = np.sum(gamma[obs_seq == l], axis=0) / np.sum(gamma, axis=0) 
            
        return prob_O, alpha, beta, gamma

# Streamlit UI Setup
st.set_page_config(page_title="HMM Baum-Welch", layout="wide", initial_sidebar_state="expanded")

# --- GLOBAL CSS INJECTION (This forces center alignment everywhere) ---
st.markdown("""
    <style>
    .stTable td, .stTable th {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Baum-Welch Algorithm Visualizer")

with st.sidebar:
    st.header("Configuration")
    states_in = st.text_input("Hidden States", "Bull Bear").split()
    symbols_in = st.text_input("Observations", "Up Down").split()
    obs_in = st.text_input("Sequence", "Up Up Down Up Down").split()
    iters = st.slider("Iterations", 1, 50, 15)

if st.button("Train Model"):
    hmm = HMM_Streamlit(states_in, symbols_in)
    sym_map = {s: i for i, s in enumerate(symbols_in)}
    obs_seq = np.array([sym_map[o] for i, o in enumerate(obs_in)])
    
    history_p = []
    
    # Training Loop
    for i in range(iters):
        p_o, alpha, beta, gamma = hmm.run_iteration(obs_seq)
        history_p.append(p_o)
        
        with st.expander(f"Iteration {i+1} Tables (Alpha, Beta, Gamma)"):
            st.write("**Forward Variable (α) - Prefix Evidence**")
            st.table(pd.DataFrame(alpha, columns=states_in))
            
            st.write("**Backward Variable (β) - Suffix Evidence**")
            st.table(pd.DataFrame(beta, columns=states_in))
            
            st.write("**State Responsibility (γ)**")
            st.table(pd.DataFrame(gamma, columns=states_in))

    # 1. Final Parameter Display
    st.header("Final Parameters (λ)")
    
    st.subheader("1. Initial Probabilities (π)")
    st.table(pd.DataFrame([hmm.pi], columns=states_in))
    
    st.subheader("2. Transition Matrix (A)")
    st.table(pd.DataFrame(hmm.A, index=states_in, columns=states_in))
    
    st.subheader("3. Emission Matrix (B)")
    st.table(pd.DataFrame(hmm.B, index=states_in, columns=symbols_in))

    # 2. Final Likelihood P(O|lambda)
    st.divider()
    final_l = history_p[-1]
    st.metric(label="Final Probability P(O|λ)", value=f"{final_l:.8f}", delta=f"{final_l - history_p[0]:.8f}")
    st.info(f"The model explains the sequence with a total probability of {final_l:.8f}.")

    # 3. Dual Graphs
    st.header("Learning Curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history_p, color='green', marker='o', label="Likelihood")
    ax1.set_title("P(O|λ) Increasing (Convergence)")
    ax1.set_xlabel("Iteration")
    ax1.grid(True)
    
    ax2.plot(1 - np.array(history_p), color='red', marker='x', label="Uncertainty")
    ax2.set_title("1 - P(O|λ) Decreasing")
    ax2.set_xlabel("Iteration")
    ax2.grid(True)
    
    st.pyplot(fig)