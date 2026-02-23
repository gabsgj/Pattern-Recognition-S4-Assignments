# HMM Baum-Welch Algorithm Implementation

## 👩‍💻 Student Details
**Name:** Archana K  
**Register Number:** TCR24CS014  

---

## 📌 Project Description
This project implements the **Hidden Markov Model (HMM)** using the **Baum–Welch Algorithm** (Expectation-Maximization approach).

The program learns the optimal:
- Transition probabilities (A)
- Emission probabilities (B)
- Initial state distribution (π)

from a given observation sequence.

---

## 🎯 Objective
To estimate the parameters of an HMM such that the likelihood **P(O | λ)** is maximized over iterations.

---

## 📥 Inputs
- Observation sequence (example: `[0, 1, 2, 1, 0, 2, 1]`)
- Number of hidden states (N)
- Number of observation symbols (M)

---

## 📤 Outputs
- Transition Matrix **A**
- Emission Matrix **B**
- Initial Distribution **π**
- Likelihood values **P(O | λ)** at each iteration
- Likelihood convergence graph
- State transition diagram

---

## 📊 Visualization
The project generates:

### 1️⃣ Likelihood Convergence Graph
Shows how **P(O|λ)** increases with each iteration, indicating convergence.

📈 File: `likelihood_graph.png`

### 2️⃣ State Transition Diagram
Graph representation of state transitions using Graphviz.

🔁 File: `state_transition_diagram.png`

---

## ⚙️ Technologies Used
- Python
- NumPy
- Matplotlib
- Graphviz

---

## 🧪 How to Run

### Step 1: Install dependencies
```bash
pip install numpy matplotlib graphviz