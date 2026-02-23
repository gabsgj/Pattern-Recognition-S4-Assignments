import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from hmm_baum_welch import HMM

st.set_page_config(page_title="HMM Baum-Welch", layout="centered")

st.title("Hidden Markov Model — Baum–Welch Algorithm")

st.write("States: S0 (Rainy), S1 (Sunny)")
st.write("Observations: 0 = Walk, 1 = Shop")

sequence_input = st.text_input(
    "Enter observation sequence (comma separated 0/1)",
    "0,1,1,0,1"
)

iterations = st.slider("Number of Iterations", 1, 100, 30)

if st.button("Train HMM"):

    O = np.array([int(x.strip()) for x in sequence_input.split(",")])
    T = len(O)

    A = [[0.7, 0.3],
         [0.4, 0.6]]

    B = [[0.1, 0.9],
         [0.6, 0.4]]

    pi = [0.6, 0.4]

    # ---------------- INITIAL TABLES ----------------
    st.subheader("Initial Parameters")

    st.markdown("### Transition Matrix A")
    st.table(np.array(A))

    st.markdown("### Emission Matrix B")
    st.table(np.array(B))

    st.markdown("### Initial State Probabilities π")
    st.table(np.array(pi))

    # ---------------- TRAIN ----------------
    model = HMM(A, B, pi)
    A_new, B_new, pi_new, log_likelihoods, A_history = model.baum_welch(O, iterations)

    log_likelihoods = np.array(log_likelihoods)
    probs = np.exp(log_likelihoods)
    neg_log_likelihood = -log_likelihoods
    prob_complement = 1 - (probs ** (1 / T))
    A_history = np.array(A_history)

    # ---------------- UPDATED TABLES ----------------
    st.subheader("Updated Parameters After Training")

    st.markdown("### Updated Transition Matrix A")
    st.table(A_new)

    st.markdown("### Updated Emission Matrix B")
    st.table(B_new)

    st.markdown("### Updated Initial Probabilities π")
    st.table(pi_new)

    # ---------------- GRAPHS ----------------

    st.subheader("Optimization Loss — Negative Log-Likelihood")
    fig1, ax1 = plt.subplots(figsize=(6,3))
    ax1.plot(neg_log_likelihood, color="red")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("NLL (-log P)")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Probability Complement — 1 - P(O|λ)^(1/T)")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.plot(prob_complement, color="purple")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("1 - P(O|λ)^(1/T)")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("Log-Likelihood Convergence")
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.plot(log_likelihoods)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Log Likelihood")
    ax3.grid(True)
    st.pyplot(fig3)

    st.subheader("Observation Probability P(O | λ)")
    fig4, ax4 = plt.subplots(figsize=(6,3))
    ax4.plot(probs)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("P(O | λ)")
    ax4.grid(True)
    st.pyplot(fig4)

    # -------- NEW PARAMETER EVOLUTION GRAPH --------
    st.subheader("Parameter Evolution — A[i][j] over Iterations")

    fig5, ax5 = plt.subplots(figsize=(7,4))
    ax5.plot(A_history[:,0,0], label="A[0][0]")
    ax5.plot(A_history[:,0,1], label="A[0][1]")
    ax5.plot(A_history[:,1,0], label="A[1][0]")
    ax5.plot(A_history[:,1,1], label="A[1][1]")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Probability")
    ax5.legend()
    ax5.grid(True)
    st.pyplot(fig5)

    # ---------------- TRANSITION DIAGRAM ----------------
    st.subheader("State Transition Diagram")

    fig, ax = plt.subplots(figsize=(6,4))

    s0 = (-1,0)
    s1 = (1,0)

    ax.add_patch(Circle(s0,0.35,color="orange",ec="black"))
    ax.add_patch(Circle(s1,0.35,color="skyblue",ec="black"))

    ax.text(-1,0,"S0",ha="center",va="center",fontsize=14,fontweight="bold")
    ax.text(1,0,"S1",ha="center",va="center",fontsize=14,fontweight="bold")

    a00 = round(A_new[0][0],2)
    a01 = round(A_new[0][1],2)
    a10 = round(A_new[1][0],2)
    a11 = round(A_new[1][1],2)

    loop0 = FancyArrowPatch((-1,0.35),(-1,0.35),
                            connectionstyle="arc3,rad=2.8",
                            arrowstyle="->",
                            mutation_scale=15)
    ax.add_patch(loop0)
    ax.text(-1,1.1,str(a00),ha="center")

    loop1 = FancyArrowPatch((1,0.35),(1,0.35),
                            connectionstyle="arc3,rad=2.8",
                            arrowstyle="->",
                            mutation_scale=15)
    ax.add_patch(loop1)
    ax.text(1,1.1,str(a11),ha="center")

    arrow01 = FancyArrowPatch((-0.65,0.2),(0.65,0.2),
                              connectionstyle="arc3,rad=0.6",
                              arrowstyle="->",
                              mutation_scale=15)
    ax.add_patch(arrow01)
    ax.text(0,0.8,str(a01),ha="center")

    arrow10 = FancyArrowPatch((0.65,-0.2),(-0.65,-0.2),
                              connectionstyle="arc3,rad=0.6",
                              arrowstyle="->",
                              mutation_scale=15)
    ax.add_patch(arrow10)
    ax.text(0,-0.9,str(a10),ha="center")

    ax.set_xlim(-2,2)
    ax.set_ylim(-1.5,1.5)
    ax.axis("off")

    st.pyplot(fig)

    st.success("Training Completed Successfully!")