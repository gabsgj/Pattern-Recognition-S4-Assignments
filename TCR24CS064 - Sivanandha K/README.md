# Hidden Markov Model (HMM) - Baum Welch Algorithm

## Student Details
Name: **SIVANANDHA K**  
Register Number: **TCR24CS064**  
Course: Pattern Recognition   

---

## 📌 Project Description

This project implements a Hidden Markov Model (HMM) using the Baum–Welch Algorithm (Expectation-Maximization approach).

The model learns the parameters of an HMM from an observed sequence.

The program computes:

- Probability of observation sequence P(O | λ)
- Initial distribution (π)
- Transition matrix (A)
- Emission matrix (B)
- Log-likelihood over iterations
- 1 − P(O | λ) graph
- State transition diagram visualization

---

## 📥 Inputs

1. Observation sequence (comma separated integers)
   Example: 0,1,1,0,1

2. Number of hidden states (N)

3. Number of observation symbols (M)

---

## 📤 Outputs

- Learned Initial Distribution (π)
- Learned Transition Matrix (A)
- Learned Emission Matrix (B)
- Probability P(O | λ)
- Log Likelihood vs Iteration Graph
- 1 − P(O | λ) vs Iteration Graph
- State Transition Diagram

---

## 🛠️ Technologies Used

- Python
- NumPy
- Streamlit
- Matplotlib
- NetworkX

---

## ⚙️ How to Run the Project

### Step 1: Install Python
Download Python from:
https://www.python.org/

Make sure Python is added to PATH.

---

### Step 2: Install Required Libraries

Open terminal in the project folder and run:

pip install -r requirements.txt

---

### Step 3: Run the Application

In the project folder, run:

streamlit run app.py

---

### Step 4: Open Browser

A browser window will automatically open.
If not, copy the local URL shown in terminal.

Click **Train Model** to train the HMM.

---

## 📊 Visualization

The application visualizes:

- Convergence of log-likelihood over iterations
- 1 − P(O | λ) trend
- State transition diagram with probabilities

---

## 📖 Algorithm Used

Baum–Welch Algorithm (Expectation-Maximization for HMM)

Steps:
1. Initialize parameters randomly
2. Compute forward probabilities (α)
3. Compute backward probabilities (β)
4. Compute γ (state responsibility)
5. Compute ξ (transition responsibility)
6. Update π, A, and B
7. Repeat until convergence

---

## 📁 Project Structure

```
HMM_BaumWelch/
│
├── hmm.py # Baum-Welch implementation
├── app.py # Streamlit visualization app
├── requirements.txt # Required libraries
└── README.md # Project documentation
```

---

## 🔓 Repository Status

This repository is public as required by assignment instructions.
