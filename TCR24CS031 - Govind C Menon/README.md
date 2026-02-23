# Hidden Markov Model using Baum-Welch Algorithm

**Name:** GOVIND C MENON  
**University Registration Number:** TCR24CS031

---

## Description

This project implements a Hidden Markov Model (HMM) trained using the Baum-Welch algorithm (Expectation-Maximization) for unsupervised parameter estimation from observation sequences.

---

## Core Implementation (`hmm_baum_welch.py`)

- **Forward Algorithm** — Computes α_t(i) = P(O₁…Oₜ, qₜ=Sᵢ | λ)
- **Backward Algorithm** — Computes β_t(i) = P(O_{t+1}…O_T | qₜ=Sᵢ, λ)
- **Gamma (γ)** — State occupancy probabilities at each time step
- **Xi (ξ)** — Joint state transition probabilities at each time step
- **Baum-Welch EM Re-estimation** — Iteratively updates A, B, π until convergence

---

## Numerical Stability Improvements

The implementation includes the following guards to prevent NaN and inf during training:

| Location | Fix | Reason |
|---|---|---|
| Log-likelihood | Epsilon clamped to 1e-300 | Prevents -inf on near-zero probabilities |
| Gamma normalization | Zero rows clamped before division | Prevents NaN when a row sums to zero |
| A matrix denominator | Zero values clamped before division | Prevents NaN in updated transition probabilities |
| B matrix denominator | Zero values clamped before division | Prevents NaN in updated emission probabilities |

---

## Web App (`app.py`)

- Configurable hidden states (N), observation symbols (M), max iterations, and tolerance
- **Live metrics** — iteration count, final log-likelihood, convergence status, P(O|λ)
- **Charts** — Log-likelihood convergence curve, |Δ log-likelihood| convergence rate
- **State Transition Diagram** — Visual graph of states and transition probabilities
- **Heatmaps** — Transition matrix A and Emission matrix B
- **Intermediate Variables** — Alpha (α), Beta (β), Gamma (γ) tables from final iteration
- **Final Learned Parameters** — Summary of converged A, B, π

---

## How to Run

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the web app:**

```bash
python app.py
```

**3. Open your browser and go to:**

```
http://127.0.0.1:5000
```

Fill in your observed sequence, number of hidden states, and other parameters, then click **Run Baum-Welch** to see the results and visualizations.

---

## Files

| File | Description |
|---|---|
| `hmm_baum_welch.py` | HMM class with Forward, Backward, and Baum-Welch implementation (CLI version) |
| `app.py` | Flask web app for interactive input and visualization |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Dependencies

- **numpy** — Matrix operations and probability computations
- **matplotlib** — Charts and state transition diagram
- **flask** — Interactive web interface
