import streamlit as st
import numpy as np
from hmm import HMM
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Hidden Markov Model - Baum-Welch Algorithm")

# Number of states and observations
states = 2
observations = 2

# Random initialization of HMM
A = np.random.dirichlet(np.ones(states), size=states)
B = np.random.dirichlet(np.ones(observations), size=states)
pi = np.random.dirichlet(np.ones(states))

model = HMM(A, B, pi)

# Input observation sequence
sequence_input = st.text_input(
    "Enter observation sequence (comma-separated, e.g., 0,1,1,0):", "0,1,1,0"
)

if st.button("Train Model"):
    try:
        O = np.array([int(x) for x in sequence_input.split(",")])
    except:
        st.error("Invalid input! Use numbers separated by commas like 0,1,1,0")
        st.stop()

    # Train HMM
    log_likelihoods = model.baum_welch(O, iterations=20)

    # Display matrices
    st.subheader("Transition Matrix (A)")
    st.write(model.A)

    st.subheader("Emission Matrix (B)")
    st.write(model.B)

    st.subheader("Initial Probabilities (pi)")
    st.write(model.pi)

    # Heatmap for A
    fig1, ax1 = plt.subplots()
    sns.heatmap(model.A, annot=True, cmap="Blues", ax=ax1)
    plt.tight_layout()
    st.pyplot(fig1)

    # Heatmap for B
    fig2, ax2 = plt.subplots()
    sns.heatmap(model.B, annot=True, cmap="Greens", ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

    # Log-likelihood convergence
    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, len(log_likelihoods)+1), log_likelihoods, marker='o')
    ax3.set_title("Log-Likelihood Convergence")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Log-Likelihood")
    st.pyplot(fig3)

    st.success("Training complete!")