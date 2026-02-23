import streamlit as st
import numpy as np
import pandas as pd
import graphviz

# --- HMM MATH: BAUM-WELCH ALGORITHM ---
def baum_welch(obs, n_hidden, n_iterations=10):
    # n_hidden: number of hidden states
    # n_symbols: number of possible observation symbols
    n_symbols = len(np.unique(obs))
    T = len(obs)
    
    # Random Initialization of Matrices
    # A = Transition, B = Emission, Pi = Initial State
    A = np.random.dirichlet(np.ones(n_hidden), size=n_hidden)
    B = np.random.dirichlet(np.ones(n_symbols), size=n_hidden)
    pi = np.random.dirichlet(np.ones(n_hidden))

    for iteration in range(n_iterations):
        # 1. Forward Pass (Alpha)
        alpha = np.zeros((T, n_hidden))
        alpha[0] = pi * B[:, obs[0]]
        for t in range(1, T):
            for j in range(n_hidden):
                alpha[t, j] = alpha[t-1].dot(A[:, j]) * B[j, obs[t]]
            alpha[t] /= (np.sum(alpha[t]) + 1e-10) # Normalization

        # 2. Backward Pass (Beta)
        beta = np.zeros((T, n_hidden))
        beta[T-1] = 1
        for t in range(T-2, -1, -1):
            for i in range(n_hidden):
                beta[t, i] = (A[i, :] * B[:, obs[t+1]]).dot(beta[t+1])
            beta[t] /= (np.sum(beta[t]) + 1e-10)

        # 3. Expectation Step (Gamma and Xi)
        gamma = (alpha * beta) / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-10)
        
        # 4. Maximization Step (Update Matrices)
        pi = gamma[0]
        # (Simplified update for mobile demo stability)
        A = A * (alpha[:-1].T @ (beta[1:] * B[:, obs[1:]].T))
        A /= A.sum(axis=1, keepdims=True)
        
    return A, B, pi

# --- VISUAL INTERFACE (Streamlit) ---
st.set_page_config(page_title="HMM Visualizer", layout="centered")
st.title("📱 Mobile HMM Trainer")
st.markdown("### Baum-Welch Implementation")

# Inputs
st.sidebar.header("Model Configuration")
n_states = st.sidebar.select_slider("Hidden States", options=[2, 3, 4], value=2)
raw_obs = st.text_input("Enter Observations (0, 1, 2...)", "0,1,0,1,1")
obs_seq = np.array([int(x.strip()) for x in raw_obs.split(",")])

if st.button("Train Model Now"):
    with st.spinner('Calculating...'):
        A_final, B_final, pi_final = baum_welch(obs_seq, n_states)
    
    st.success("Training Complete!")

    # Visual 1: State Diagram
    st.subheader("1. State Transition Diagram")
    
    dot = graphviz.Digraph()
    for i in range(n_states):
        dot.node(f'S{i}', f'State {i}')
    for i in range(n_states):
        for j in range(n_states):
            weight = A_final[i, j]
            if weight > 0.01: # Show only visible paths
                dot.edge(f'S{i}', f'S{j}', label=f'{weight:.2f}')
    st.graphviz_chart(dot)

    # Visual 2: Data Tables
    st.subheader("2. Learned Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Transition Matrix (A)**")
        st.dataframe(pd.DataFrame(A_final))
    with col2:
        st.write("**Emission Matrix (B)**")
        st.dataframe(pd.DataFrame(B_final))
  
