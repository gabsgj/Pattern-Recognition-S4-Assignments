# Hidden Markov Model using Baum-Welch Algorithm

Name: KASINATHAN A B
University Registration Number: TCR24CS042

## Description

This project implements a **Hidden Markov Model (HMM)** trained using the **Baum-Welch algorithm** (Expectation-Maximization) for unsupervised parameter estimation from observation sequences.

### Core Implementation (`hmm.py`)

- **Forward Algorithm** — Computes α_t(i) = P(O₁…Oₜ, qₜ=Sᵢ | λ)
- **Backward Algorithm** — Computes β_t(i) = P(O_{t+1}…O_T | qₜ=Sᵢ, λ)
- **Gamma (γ)** — State occupancy probabilities at each time step
- **Xi (ξ)** — Joint state transition probabilities at each time step
- **Baum-Welch EM Re-estimation** — Iteratively updates A, B, π until convergence

#### Numerical Stability Improvements

| Location | Fix | Reason |
|----------|-----|--------|
| Log-likelihood | Epsilon raised from `1e-10` → `1e-300` | `1e-10` pollutes log values on short sequences |
| Gamma normalization | Zero rows clamped before division | Prevents `NaN` when a row sums to zero |
| A matrix denominator | Zero values clamped before division | Prevents `NaN` in updated transition probabilities |
| B matrix denominator | Zero values clamped before division | Prevents `NaN` in updated emission probabilities |

### Web App (`app.py`)

- Configurable hidden states (N), observation symbols (M), max iterations, and tolerance
- Live metrics — iteration count, log-likelihood, Δ change, convergence status
- **Charts** — Log-likelihood convergence, observation probability P(O|λ), negative log-likelihood loss
- **State Transition Diagram** — Visual graph of states and emission connections
- **Intermediate Variables** — Alpha, Beta, Gamma tables (final iteration)
- **Final Learned Parameters** — Summary of converged A, B, π
- **HMM Theory Reference** — Expandable sections covering all algorithm equations

## 🚀 Live Demo

> **Deployed App:** https://hmm-baum-welch-implementation-net5sj8rryxvddwmcp6xdh.streamlit.app

## How to Run

### Option 1 — Use the deployed app
Visit the live demo link above — no installation needed.

### Option 2 — Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Files

| File | Description |
|------|-------------|
| `hmm.py` | HMM class with Forward, Backward, and Baum-Welch implementation |
| `app.py` | Streamlit web app for interactive visualization |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, and cache files from git |

## Dependencies

- `numpy` — Matrix operations and probability computations
- `streamlit` — Interactive web interface
- `matplotlib` — Charts and state transition diagram
- `pandas` — Iteration log and parameter tables


## Live Deployment
[Live Link](https://hmm-baum-welch-implementation-net5sj8rryxvddwmcp6xdh.streamlit.app/)
