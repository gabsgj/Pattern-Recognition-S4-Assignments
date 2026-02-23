import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmm import baum_welch

st.title("HMM - Baum Welch Algorithm")

obs_input = st.text_input("Observation Sequence (comma separated)", "0,1,1,0,1")
hidden_states = st.number_input("Number of Hidden States", min_value=2, value=2)
symbols = st.number_input("Number of Observation Symbols", min_value=2, value=2)

if st.button("Train Model"):

    O = np.array([int(x.strip()) for x in obs_input.split(",")])
    N = int(hidden_states)
    M = int(symbols)

    pi, A, B, P, logs = baum_welch(O, N, M)

    st.subheader("P(O | λ)")
    st.write(float(P))

    st.subheader("Initial Distribution (π)")
    st.write(pi.tolist())

    st.subheader("Transition Matrix (A)")
    st.write(A.tolist())

    st.subheader("Emission Matrix (B)")
    st.write(B.tolist())

    # -------- Log Likelihood Graph --------
    st.subheader("Log Likelihood vs Iteration")
    plt.clf()
    plt.figure()
    plt.plot(logs)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    # -------- 1 - P(O|λ) Graph --------
    st.subheader("1 - P(O | λ) vs Iteration")
    plt.clf()
    plt.figure()
    prob_values = [np.exp(l) for l in logs]
    plt.plot([1 - p for p in prob_values])
    plt.xlabel("Iteration")
    plt.ylabel("1 - P(O | λ)")
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    # -------- State Transition Diagram --------
    st.subheader("State Transition Diagram")
    plt.clf()
    G = nx.DiGraph()

    for i in range(N):
        for j in range(N):
            G.add_edge(f"S{i}", f"S{j}", weight=round(A[i][j], 2))

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Validation Check --------
    st.subheader("Probability Validation Check")
    st.write("Sum of π:", float(np.sum(pi)))
    st.write("Row sums of A:", A.sum(axis=1).tolist())
    st.write("Row sums of B:", B.sum(axis=1).tolist())
