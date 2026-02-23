import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmm import HMM

st.title("Hidden Markov Model - Baum Welch Visualizer")

st.write("Enter observation sequence (example: 0,1,0,2,1,0)")

obs_input = st.text_input("Observation Sequence")

num_states = st.slider("Number of Hidden States", 2, 5, 2)
num_obs = st.slider("Number of Observation Symbols", 2, 5, 3)

iterations = st.slider("Training Iterations", 5, 50, 20)

if st.button("Train Model"):

    try:
        obs = np.array([int(x.strip()) for x in obs_input.split(",")])

        model = HMM(n_states=num_states, n_obs=num_obs)

        model.baum_welch(obs, n_iter=iterations)

        st.success("Training Completed")

        st.subheader("Initial Probabilities (π)")
        st.write(model.pi)

        st.subheader("Transition Matrix (A)")
        st.write(model.A)

        st.subheader("Emission Matrix (B)")
        st.write(model.B)

        # Heatmap of transition matrix
        st.subheader("Transition Matrix Heatmap")

        fig1, ax1 = plt.subplots()
        cax = ax1.imshow(model.A)
        ax1.set_title("State Transition Probabilities")
        ax1.set_xlabel("To State")
        ax1.set_ylabel("From State")
        fig1.colorbar(cax)

        st.pyplot(fig1)

        # STATE TRANSITION DIAGRAM
        st.subheader("State Transition Diagram")

        G = nx.DiGraph()

        states = [f"S{i}" for i in range(num_states)]

        for s in states:
            G.add_node(s)

        for i in range(num_states):
            for j in range(num_states):
                prob = model.A[i][j]
                if prob > 0.01:
                    G.add_edge(states[i], states[j], weight=round(prob, 2))

        pos = nx.circular_layout(G)

        fig2, ax2 = plt.subplots()

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3000,
            font_size=12,
            ax=ax2
        )

        edge_labels = nx.get_edge_attributes(G, "weight")

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax2
        )

        ax2.set_title("HMM State Transition Diagram")

        st.pyplot(fig2)

    except:
        st.error("Invalid input. Example: 0,1,2,1,0")