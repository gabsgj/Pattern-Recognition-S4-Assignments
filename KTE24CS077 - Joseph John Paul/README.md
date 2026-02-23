# Baum-Welch Algorithm – Hidden Markov Model Training

> **Joseph John Paul**  
> University Registration No.: KTE24CS077  
> Admission No.: 24B1221

---

## About the Project

This project implements the **Baum-Welch Algorithm** from scratch in Python — an Expectation-Maximization (EM) algorithm used to train **Hidden Markov Models (HMMs)**. Given an observation sequence, the algorithm iteratively updates the model parameters (initial state probabilities `π`, transition matrix `A`, and emission matrix `B`) to maximize the probability of the observed sequence.

The implementation also includes **convergence visualization**, plotting how `1 - P(O|λ)` decreases at each iteration as the model improves.

---

## Files

| File | Description |
|------|-------------|
| `baum_welch.py` | Main implementation of the Baum-Welch algorithm with visualization |
| `baum_welch_convergence.png` | Convergence plot generated after running the program |
| `README.md` | Project documentation |

---

## Requirements

Make sure you have **Python 3.7+** installed. Then install the required libraries:

```bash
pip install numpy matplotlib
```

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations and numerical computation |
| `matplotlib` | Plotting the convergence graph |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

**2. Install dependencies**
```bash
pip install numpy matplotlib
```

**3. Run the program**
```bash
python baum_welch.py
```

**4. Enter inputs when prompted**

The program will ask for two inputs:

- **Observation sequence** — A string of `0`s and `1`s (binary observations), e.g.:
  ```
  Enter observation sequence: 010011101
  ```

- **Number of hidden states** — A positive integer, e.g.:
  ```
  Enter no of states: 2
  ```

**5. Output**

- The terminal will print `P(O|λ)` and `1 - P(O|λ)` at each iteration until convergence.
- Final values of `π`, `A`, and `B` are printed at the end.
- A convergence plot (`baum_welch_convergence.png`) is saved in the current directory and displayed automatically.

---

## Example Run

```
Enter observation sequence: 01101001
Enter no of states: 2

Iteration 1 | P(O|λ) = 0.00724831 | 1 - P(O|λ) = 0.99275169
Iteration 2 | P(O|λ) = 0.00731204 | 1 - P(O|λ) = 0.99268796
...
Iteration N | P(O|λ) = 0.00741000 | 1 - P(O|λ) = 0.99259000

Initial state probabilities: [0.612 0.388]
Transition probabilities: [[0.543 0.457] [0.381 0.619]]
Emission probabilities:    [[0.489 0.511] [0.402 0.598]]
Probability of the observation sequence: 0.00741
```

---

## How It Works

1. **Initialization** — `π`, `A`, and `B` are randomly initialized and normalized.
2. **Forward pass (α)** — Computes the probability of the observation sequence up to time `t`.
3. **Backward pass (β)** — Computes the probability of future observations from time `t`.
4. **E-step** — Computes `γ` (state occupancy) and `η` (transition occupancy) using `α` and `β`.
5. **M-step** — Re-estimates `π`, `A`, and `B` to maximize `P(O|λ)`.
6. Steps 2–5 repeat until the change in `P(O|λ)` is less than the tolerance `1e-7`.

---

## Convergence Plot

After convergence, the program generates a two-panel plot:

- **Left panel** — `1 - P(O|λ)` vs iteration (drops toward 0 as model improves)
- **Right panel** — `P(O|λ)` vs iteration (rises toward its maximum)

---

## Note

- The observation sequence must contain only `0`s and `1`s (binary alphabet).
- Results may vary between runs due to random initialization.
- Increase the number of states for more complex observation patterns.
