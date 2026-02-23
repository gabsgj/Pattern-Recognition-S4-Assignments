# HMM_baum_welch
# Hidden Markov Model – Baum Welch Algorithm

Jany sabarinath 
TCR24CS035 

---

## Description

This project implements the Hidden Markov Model (HMM) using the Baum–Welch Algorithm in Python.

Given:
- Observed state sequence
- Number of hidden states

The program computes:
- Transition Matrix (A)
- Emission Matrix (B)
- Initial Distribution (π)
- Probability P(O | λ)

Likelihood progression is visualized using a graph.

---

## Requirements

Install dependencies:

pip install numpy matplotlib

---

## How to Run

python hmm_baum_welch.py

---

## Inputs

Modify inside the script:

observed_sequence = [0, 1, 0, 2, 1]
n_hidden_states = 2
n_observation_symbols = 3