# Hidden Markov Model using Baum–Welch Algorithm

## Name: Anna Tomy  
## University Registration Number: TCR24CS013 

---

## Project Description

This project demonstrates training of a Hidden Markov Model (HMM) using the Baum-Welch algorithm, which is a special case of the Expectation–Maximization (EM) algorithm.

Given an observation sequence, the model learns:

- Transition probability matrix (A)
- Emission probability matrix (B)
- Initial state distribution (π)
- Log-likelihood progression
- Visual state transition diagram

The application is implemented as an interactive web app using Streamlit.

---

## 🧠 Hidden Markov Model (HMM)

A Hidden Markov Model is a statistical model where:

- The system is assumed to be a Markov process with hidden states
- Only observations are visible
- The underlying state sequence is unknown

HMMs are widely used in:

- Speech recognition
- Natural language processing
- Bioinformatics
- Pattern recognition
- Time-series analysis

---

## ⚙️ Baum-Welch Algorithm

The Baum-Welch algorithm estimates unknown HMM parameters from data.

It iteratively performs:

1. **Expectation (E-step)** — compute probabilities of hidden states  
2. **Maximization (M-step)** — update parameters to maximize likelihood  

The process repeats until convergence.

---

## 🖥️ Features of the Application

✔ User-defined number of hidden states  
✔ User-defined observation symbols  
✔ Custom observation sequence input  
✔ Adjustable iterations and tolerance  
✔ Displays learned parameters (A, B, π)  
✔ Log-likelihood output  
✔ Automatic HMM state diagram generation  

---

## 🧩 Technologies Used

- Python 3
- Streamlit
- NumPy
- Graphviz (for state diagram visualization)

---

Deployed Application:
https://hmm-baum-welch-qfj9s2bqkp8xtzhuphfyaa.streamlit.app/
