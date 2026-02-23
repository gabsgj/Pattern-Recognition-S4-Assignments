# Hidden Markov Model – Baum–Welch Algorithm Implementation

## Student Information

**Name:** Alan Ali

**Register Number:** TCR24CS106

**Course:** Pattern Recognition

---

## Project Description

This project implements the Baum–Welch algorithm for training a Hidden Markov Model (HMM).

Baum–Welch is a special case of the Expectation–Maximization (EM) algorithm used to estimate HMM parameters when only the observation sequence is available.

All computations are implemented manually in Python without using external HMM libraries.

The project also includes an interactive Streamlit interface for training and visualization.

---

## Hidden Markov Model Definition

An HMM is defined by the parameter set:

```
λ = (A, B, π)
```

Where:

* **A** = State Transition Probability Matrix
* **B** = Emission Probability Matrix
* **π** = Initial State Distribution

The goal of Baum–Welch is to estimate these parameters such that the likelihood `P(O | λ)` is maximized.

---

## Algorithm Components Implemented

### 1. Forward Algorithm (α)

Computes:

```
α_t(i) = P(O1, O2, ..., Ot, qt = i | λ)
```

Scaling is applied to prevent numerical underflow.

---

### 2. Backward Algorithm (β)

Computes:

```
β_t(i) = P(O_{t+1}, ..., OT | qt = i, λ)
```

---

### 3. State Responsibility (γ)

```
γ_t(i) = P(qt = i | O, λ)
```

Represents how responsible hidden state *i* is for observation at time *t*.

---

### 4. Transition Responsibility (ξ)

```
ξ_t(i,j) = P(qt = i, qt+1 = j | O, λ)
```

Represents expected transitions from state *i* to state *j*.

---

### 5. Parameter Re-estimation

Initial distribution:

```
π_i = γ_1(i)
```

Transition matrix:

```
a_ij = (sum of ξ_t(i,j)) / (sum of γ_t(i))
```

Emission matrix:

```
b_i(k) = (sum of γ_t(i) where observation = k) / (sum of γ_t(i))
```

These updates are repeated until convergence.

---

## Inputs

The program allows configuration of:

* Observed state sequence (comma-separated integers)
* Number of hidden states
* Maximum number of iterations
* Convergence tolerance

The user may provide manual input or generate a synthetic sequence.

---

## Outputs

After training, the application displays:

* Initial Transition Matrix (A)
* Final Transition Matrix (A)
* Initial Emission Matrix (B)
* Final Emission Matrix (B)
* Initial Distribution (π)
* Final Distribution (π)
* Final Log-Likelihood `P(O | λ)`
* Convergence information

---

## Visualization

The application provides:

* Log-likelihood vs iteration graph
* Learned state transition diagram
* Matrix values in tabular format

The likelihood increases monotonically until convergence, validating correct EM implementation.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Streamlit
* Graphviz

---

## Project Structure

```
HMM-Baum-Welch/
│
├── baum_welch.py
├── diagram_generator.py
├── app.py
├── requirements.txt
├── README.md
```

---

## How to Run the Project

### 1. Clone Repository

```
git clone https://github.com/AlanAli-byte/HMM-Baum-Welch.git
cd HMM-Baum-Welch
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Install Graphviz

Download from:

[https://graphviz.org/download/](https://graphviz.org/download/)

Ensure Graphviz is added to system PATH.

### 4. Run Application

```
streamlit run app.py
```

---

## Important Notes

* The Baum–Welch algorithm is implemented from scratch.
* No external HMM libraries are used.
* Scaling is applied to prevent numerical instability.
* Log-likelihood increases across iterations until convergence.
* Repository is public as required.

---

## Conclusion

This project demonstrates a complete implementation of the Baum–Welch algorithm along with visualization tools to understand convergence behavior and learned hidden state transitions.

---
