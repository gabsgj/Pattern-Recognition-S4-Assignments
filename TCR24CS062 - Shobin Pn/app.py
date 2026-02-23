import streamlit as st
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmm import train_hmm

st.title("HMM - Baum Welch Algorithm")

# User Inputs
obs_input = st.text_input(
    "Enter observations (comma separated numbers):",
    "0,1,1,0,1"
)

num_states = st.number_input(
    "Number of hidden states:",
    min_value=2,
    value=2
)

# Function to draw transition diagram
def draw_transition_diagram(A):
    dot = graphviz.Digraph()
    n_states = len(A)

    for i in range(n_states):
        dot.node(f"S{i}", f"State {i}")

    for i in range(n_states):
        for j in range(n_states):
            prob = round(float(A[i][j]), 3)
            dot.edge(f"S{i}", f"S{j}", label=str(prob))

    return dot


# Train button
if st.button("Train Model"):

    # Convert observations
    observations = np.array([int(x.strip()) for x in obs_input.split(",")])
    M = len(set(observations))

    # Train HMM
    A, B, pi, likelihoods, gamma = train_hmm(
        observations,
        num_states,
        M
    )

    # Show π
    st.subheader("Initial Distribution (π)")
    st.write(pi)

    # Show A
    st.subheader("Transition Matrix (A)")
    st.write(A)

    # Show B
    st.subheader("Emission Matrix (B)")
    st.write(B)

    # Likelihood Graph
    st.subheader("Likelihood Convergence")
    plt.figure()
    plt.plot(likelihoods)
    plt.xlabel("Iterations")
    plt.ylabel("P(O | λ)")
    plt.title("Likelihood Convergence")
    st.pyplot(plt)

    # Transition Diagram
    st.subheader("State Transition Diagram")
    diagram = draw_transition_diagram(A)
    st.graphviz_chart(diagram)

    # Gamma Table
    st.subheader("Hidden State Probabilities (Gamma)")

    gamma = np.array(gamma)
    gamma_table = []

    T = len(observations)

    # Detect gamma shape automatically
    if gamma.shape[0] == T:
        # gamma shape is (T, num_states)
        for t in range(T):
            row = {}
            for i in range(num_states):
                row[f"State {i}"] = round(float(gamma[t][i]), 4)
            gamma_table.append(row)
    else:
        # gamma shape is (num_states, T)
        for t in range(T):
            row = {}
            for i in range(num_states):
                row[f"State {i}"] = round(float(gamma[i][t]), 4)
            gamma_table.append(row)

    df = pd.DataFrame(gamma_table)
    st.dataframe(df)