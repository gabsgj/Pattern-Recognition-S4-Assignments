import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from baum_welch import baum_welch

st.set_page_config(page_title="HMM - Baum Welch", layout="centered")

st.title("Hidden Markov Model - Baum Welch Algorithm")

sequence_input = st.text_area(
    "Observed State Sequence (comma separated numbers)",
    "0,1,0,2,1,0"
)

N = st.number_input("Number of Hidden States", min_value=1, value=2)

max_iter = st.number_input("Max Iterations", min_value=1, value=50)

tol = st.number_input("Convergence Tolerance", value=0.0001, format="%.6f")

if st.button("Run Training"):

    O = np.array([int(x.strip()) for x in sequence_input.split(",")])

    A, B, pi, log_likelihoods = baum_welch(O, N, max_iter, tol)

    st.subheader("Initial Distribution π")
    st.dataframe(pi)

    st.subheader("Transition Matrix A")
    st.dataframe(A)

    st.subheader("Emission Matrix B")
    st.dataframe(B)

    st.subheader("Log Likelihood over Iterations")

    fig, ax = plt.subplots()
    ax.plot(log_likelihoods)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log P(O | λ)")
    st.pyplot(fig)

    st.subheader("State Transition Diagram")

    dot = Digraph()

    for i in range(N):
        dot.node(f"S{i}", f"S{i}")

    for i in range(N):
        for j in range(N):
            if A[i, j] > 0.01:
                dot.edge(f"S{i}", f"S{j}", label=f"{A[i,j]:.2f}")

    st.graphviz_chart(dot)