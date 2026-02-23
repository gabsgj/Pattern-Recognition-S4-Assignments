import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmm import HMM

st.set_page_config(page_title="HMM Baum-Welch Visualizer", layout="wide")

st.title("Hidden Markov Model using Baum-Welch Algorithm")

st.write("Enter observation sequence (comma separated numbers like 0,1,2,0,1):")

obs_input = st.text_input("Observation Sequence")

n_states = st.number_input("Number of Hidden States", min_value=2, max_value=5, value=2)
n_obs = st.number_input("Number of Observation Symbols", min_value=2, max_value=5, value=3)
iterations = st.slider("Training Iterations", 5, 50, 20)

if st.button("Train HMM"):

    try:
        obs = np.array([int(x.strip()) for x in obs_input.split(",")])

        model = HMM(n_states=n_states, n_obs=n_obs)
        model.baum_welch(obs, n_iter=iterations)

        st.success("Training Completed Successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transition Matrix (A)")
            st.write(model.A)

            st.subheader("Initial Probabilities (π)")
            st.write(model.pi)

        with col2:
            st.subheader("Emission Matrix (B)")
            st.write(model.B)

        # -------------------------
        # Transition Matrix Heatmap
        # -------------------------
        st.subheader("Transition Matrix Heatmap")

        fig1, ax1 = plt.subplots()
        cax = ax1.imshow(model.A, cmap="Blues")
        plt.colorbar(cax)
        ax1.set_title("Transition Matrix")
        ax1.set_xlabel("To State")
        ax1.set_ylabel("From State")
        st.pyplot(fig1)

        # -------------------------
        # State Transition Diagram
        # -------------------------
        st.subheader("State Transition Diagram")

        G = nx.DiGraph()

        # Add nodes
        for i in range(model.n_states):
            G.add_node(f"S{i}")

        # Add edges with probabilities
        for i in range(model.n_states):
            for j in range(model.n_states):
                prob = round(model.A[i][j], 2)
                if prob > 0.01:
                    G.add_edge(f"S{i}", f"S{j}", weight=prob)

        pos = nx.circular_layout(G)
        fig2, ax2 = plt.subplots()

        nx.draw(
            G, pos,
            with_labels=True,
            node_color="lightblue",
            node_size=3000,
            font_size=12,
            ax=ax2
        )

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2)

        st.pyplot(fig2)

        # -------------------------
        # Emission Display Per State
        # -------------------------
        st.subheader("Emission Probabilities by State")

        for state in range(model.n_states):
            st.write(f"State S{state} emits:", np.round(model.B[state], 3))

    except:
        st.error("Invalid Input! Please enter valid comma-separated numbers.")
