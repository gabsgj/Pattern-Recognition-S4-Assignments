# Hidden Markov Model — Baum-Welch Algorithm Explorer

An interactive web-based visualization of Hidden Markov Models and the Baum–Welch (EM) algorithm, built with HTML, CSS, and JavaScript.

---

## 👤 Student Details

| Field | Details |
|---|---|
| **Name** | Cilla Elsa Binoy |
| **University** | Government Engineering College Thrissur |
| **Department** | CSE |
| **Register Number** | TCR24CS021 |
| **Course** | Pattern Recognition |
| **Submission Date** | 21 February 2026 |

---

## 📁 Project Structure

```
hmm-baum-welch/
│
├── index.html       # Main HTML — all 9 sections of content
├── style.css        # Styling — Crimson & Pink dark theme
├── script.js        # JavaScript — Baum-Welch, Viterbi, canvas animations
└── README.md        # This file
```

---

## 📖 Features

- **Live State Machine** — animated canvas showing real-time state transitions
- **Forward Algorithm (α)** — step-by-step worked example with formulas
- **Backward Algorithm (β)** — backward pass with numerical example
- **Auxiliary Variables (γ, ξ)** — soft credit assignment visualization
- **Parameter Update Equations** — π, A, B re-estimation (M-step)
- **Interactive Playground** — train your own HMM with custom N, M, iterations, and observation sequences
- **Convergence Chart** — real-time log-likelihood plot across iterations
- **Viterbi Decoder** — decode the most probable hidden state sequence with trellis diagram
- **Mathematical Deep Dive** — EM convergence proof, scaling, complexity analysis

---

## 🚀 Steps to Run

### Option 1 — Open Directly in Browser (Simplest)

1. Download all three files into the **same folder**:
   - `index.html`
   - `style.css`
   - `script.js`

2. Double-click `index.html` to open it in your browser.

> ✅ No installation, no server, no dependencies required.  
> ✅ Works in Chrome, Firefox, Edge, Safari.

---

### Option 2 — Run with a Local Server (Recommended for best results)

Using **VS Code Live Server**:
1. Install the [Live Server extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) in VS Code
2. Open the project folder in VS Code
3. Right-click `index.html` → **"Open with Live Server"**
4. Browser opens automatically at `http://127.0.0.1:5500`

Using **Python** (if installed):
```bash
# Navigate to the project folder
cd path/to/hmm-baum-welch

# Python 3
python -m http.server 8000

# Then open in browser:
# http://localhost:8000
```

Using **Node.js** (if installed):
```bash
npx serve .
# Then open the URL shown in the terminal
```

---

## 🧮 Mathematical Concepts Covered

| Concept | Formula |
|---|---|
| HMM Definition | λ = (A, B, π) |
| Forward Variable | αt(i) = P(O₁…Ot, qt=i \| λ) |
| Backward Variable | βt(i) = P(Ot+1…OT \| qt=i, λ) |
| State Responsibility | γt(i) = αt(i)βt(i) / P(O\|λ) |
| Transition Responsibility | ξt(i,j) = αt(i) aij bj(Ot+1) βt+1(j) / P(O\|λ) |
| Transition Update | â_ij = Σξt(i,j) / Σγt(i) |
| Emission Update | b̂i(o) = Σ[Ot=o] γt(i) / Σγt(i) |
| Viterbi Recursion | δt+1(j) = max_i[δt(i) aij] bj(Ot+1) |

---

## 🌐 External Libraries Used

| Library | Purpose | Source |
|---|---|---|
| MathJax 3 | Rendering LaTeX math formulas | CDN |
| Google Fonts | Syne, Space Mono, JetBrains Mono | CDN |

> All other functionality (Baum-Welch, Viterbi, canvas animations, charts) is implemented from scratch in `script.js`.

---

## 📚 References

- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE, 77(2), 257–286.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
