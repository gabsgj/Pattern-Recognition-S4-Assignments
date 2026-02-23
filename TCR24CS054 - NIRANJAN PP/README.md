# HMM Baum-Welch Algorithm

## Name

NIRANJAN PP

## Register Number

TCR24CS054

## Description

This project implements the **Hidden Markov Model (HMM)** training using the **Baum-Welch algorithm**.
The program estimates:

* Transition matrix (A)
* Emission matrix (B)
* Initial distribution (π)
* Probability of observed sequence ( P(O \mid \lambda) )
* Likelihood convergence graph across iterations

## Inputs

* Observed state sequence
* Number of hidden states

## Outputs

* Transition probabilities between states
* Emission probabilities
* Initial state distribution
* Final likelihood value
* Likelihood convergence graph

## How to Run

### 1. Install dependencies

```
pip install numpy matplotlib
```

### 2. Run the program

```
python hmm.py
```

### 3. Output

The program prints matrices A, B, π and displays a likelihood convergence graph.
