# Hidden Markov Model — Baum-Welch Algorithm

## Description

Implementation of a Hidden Markov Model (HMM) trained using the **Baum-Welch Algorithm**.  
Running the Python file trains the model **and automatically opens a visual browser app** showing the results.

## Topic

Pattern Recognition — CSE S4

## How it Works

The Baum-Welch Algorithm is an EM (Expectation-Maximization) method with three steps:

1. **Forward Algorithm** — computes α (probability of partial observations ending in each state)
2. **Backward Algorithm** — computes β (probability of future observations from each state)  
3. **Parameter Re-estimation** — updates π, A, and B using γ and ξ

## HMM Setup

| Component | Values |
|-----------|--------|
| States | Rainy, Sunny |
| Observations | Walk, Shop, Clean |
| Sequence | Walk → Shop → Clean → Walk |

## Language & Requirements

- Python 3
- NumPy (`pip install numpy`)

## How to Run

```bash
pip install numpy
python hmm_baumwelch.py
```

The script will:
1. Print training progress in the terminal
2. **Automatically open the visualizer in your browser**

The browser app lets you scrub through each iteration and see how A, B, π, α, β, and log-likelihood evolve.

## Submitted By

- **Name:** Dwaraka Dileep
- **University Registration No:** LTCR24CS072
- **Subject:** Pattern Recognition
- **Assignment:** HMM using Baum-Welch Algorithm