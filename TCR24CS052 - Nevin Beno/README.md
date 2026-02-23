# Hidden Markov Model (HMM) using Baum-Welch Algorithm

## Student Details
**Name:** Nevin Beno <br>
**University Registration Number:** TCR24CS052 

---
## Project Description

This project implements a **Hidden Markov Model (HMM)** trained using the **Baum-Welch Algorithm** (Expectation-Maximization method).

The implementation is written in **Python** and includes visualizations for:

- Log-likelihood convergence
- Transition probability matrix
- Emission probability matrix

This satisfies the requirement of implementing HMM using Baum-Welch and visually presenting the results.

---
## System Architecture: 
```
.
├── README.md
├── .gitignore
├── requirements.txt
├── media/
│   ├── emission_matrix.png
│   ├── likelihood.png
│   └── transition_matrix.png
└── src.py
```
## Installation and Setup: 
1. Clone the repo: 
    ```bash
    git clone git@github.com:nevinbeno/baum-welch-algo # SSH
        # or
    git clone https://github.com/nevinbeno/baum-welch-algo # HTTPS
    ```
2. Get into the repo: 
    ```bash
    cd baum-welch-algo
    ```
3. Create Virtual Environment: 
    ```bash
    python3 -m venv venv # Linux
        # (or)
    python -m venv venv # windows
    ```
4. Activate it: 
    ```bash
    source venv/bin/activate # linux
        # or
    venv\Scripts\activate # Windows
    ```
    Output: 
    ```
    (venv) $  (linux)
    (venv) >  (Windows)
    ```
5. Install dependencies (**ONLY INSIDE venv**): 
    ```bash
    pip install --upgrade pip # (optional)
    ```
    ```bash
    pip install -r requirements.txt
    ```
6. Run: 
    ```bash
    python3 src.py
    ```
    _____
    - Enter the no. of hidden states: 
    - Enter the no. of observation symbols: 
    - Enter observation sequence (space separated integers):
    - Enter number of training iterations:
______
## Sample Output: 
```
    === Hidden Markov Model Trainer (Baum-Welch) ===
    Enter number of hidden states: 2
    Enter number of observation symbols: 3
    Enter observation sequence (space separated integers): 0 1 2 0 1 2 1 2 1 0 2 0 1 0 2 2 2 1 1 0 0 0 1 0 1 0 2 0 0 2 0 1 0
    Enter number of training iterations: 80

    Transition Matrix A:
    [[2.29969788e-06 9.99997700e-01]
    [8.14926176e-01 1.85073824e-01]]

    Emission Matrix B:
    [[0.05209091 0.61290374 0.33500535]
    [0.71126662 0.06403846 0.22469491]]

    Initial State Probabilities pi:
    [1.58563415e-98 1.00000000e+00]

    Plots saved inside 'screenshots' folder.
```
## Visualizations (Heat map form) 
### Transition matrix: 
<div align = "center">
    <image src = "media/transition_matrix.png"><br>Transition matrix.png</img>
</div><br>

- Lower colour indicates a lower value of probability for transitioning. 
- Likewise, a brighter colour indicates a higher probability of transition.
### Emission Matrix: 
<div align = "center">
    <image src = "media/emission_matrix.png"><br>Emission matrix.png</img>
</div><br>

- Dark colour indicates a lower probability of occurrence of that symbol. 
- Like wise, brighter color indicates a higher probability of occurrence of that symbol. 
### Log - likelihood convergence
<div align = "center">
    <image src = "media/likelihood.png"><br>likelihood convergence.png</img>
</div><br>

- Log - likelihood curve changes very quickly in early iterations. 
- Then improves very slowly. 
- Then becomes almost flat (plateau), indicating that the model parameters are no longer changing meaningfully. 
_______
# Theory: 

## What is a Hidden Markov Model (HMM)?

A Hidden Markov Model is a statistical model used to represent systems that:

- Have hidden (unobservable) states
- Produce observable outputs
- Follow probabilistic transitions between states

An HMM consists of:

1. **Hidden States**
2. **Observation Symbols**
3. **Transition Probability Matrix (A)**
4. **Emission Probability Matrix (B)**
5. **Initial State Probability Vector (π)**

---

## Transition Matrix (A)

The transition matrix defines the probability of moving from one hidden state to another.

Each row sums to 1.

Example interpretation:
- A[0][1] = 0.8 means if we are in State 0, there is 80% chance of moving to State 1.

---

## Emission Matrix (B)

The emission matrix defines the probability of observing a symbol given a hidden state.

Each row sums to 1.

Example interpretation:
- B[1][0] = 0.7 means when in State 1, there is 70% chance of observing symbol 0.

---

## Initial State Probabilities (π)

The initial probability vector defines the probability of starting in each hidden state.


All values sum to 1.

---

## What is the Baum-Welch Algorithm?

Baum-Welch is an **Expectation-Maximization (EM)** algorithm used to estimate HMM parameters when hidden states are unknown.

It works iteratively:

1. **Forward Algorithm** – Computes probability of observation sequence.
2. **Backward Algorithm** – Computes backward probabilities.
3. **Expectation Step (E-step)** – Computes gamma and xi.
4. **Maximization Step (M-step)** – Updates A, B, and π.
5. Repeat until convergence.

---

## Log-Likelihood Convergence

During training, the log-likelihood of the observation sequence increases monotonically.

This indicates that the model parameters are improving at each iteration.

The convergence graph is saved as a file in the `screenshots/` folder. 

____

## Conclusion

This project successfully implements:

- Hidden Markov Model
- Baum-Welch parameter estimation
- Forward-Backward algorithm with scaling
- Proper probability normalization
- Visual representation of results

The model demonstrates correct convergence behavior and valid probabilistic matrices.
