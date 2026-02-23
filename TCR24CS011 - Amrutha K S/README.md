# HMM - Baum Welch Algorithm Implementation

## 👩‍💻 Student Details
Name: Amrutha K S  
Register Number: TCR24CS011 

---

## 📘 Project Description

This project implements the Baum-Welch Algorithm (Expectation–Maximization algorithm) for training a Hidden Markov Model (HMM).

The program:

- Initializes random transition, emission, and initial probability matrices
- Computes Forward and Backward probabilities
- Calculates Gamma and Xi values
- Re-estimates model parameters iteratively
- Plots likelihood convergence graph
- Outputs final trained HMM parameters

---

## 📥 Inputs

- Observation sequence
- Number of hidden states
- Number of observation symbols

---

## 📤 Outputs

- Transition Matrix (A)
- Emission Matrix (B)
- Initial Distribution (π)
- Probability P(O | λ)
- Likelihood values per iteration
- Convergence Graph

---

## 🛠 Requirements

Python 3.x

Required Python Libraries:

- numpy
- matplotlib

Install them using:

pip install numpy matplotlib

---

## ▶️ How to Run

1. Open terminal inside the project folder
2. Run:

python main.py

---

## 📈 Visualization

The program generates a convergence graph showing how P(O | λ) increases over iterations, demonstrating the EM algorithm behavior.

---

## 📚 Algorithm Implemented

Baum-Welch Algorithm for Hidden Markov Models (HMM)