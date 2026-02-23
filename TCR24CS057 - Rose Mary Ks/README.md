# Hidden Markov Model (HMM) using Baum–Welch Algorithm

## Student Details

**Name:** Rose Mary KS  
**Register Number:** TCR24CS057  

---

## 📌 Project Description

This project implements a **Hidden Markov Model (HMM)** using the **Baum–Welch Algorithm**, which is an Expectation–Maximization (EM) algorithm used to estimate the parameters of an HMM when only observation sequences are available.

The model trains transition and emission probabilities iteratively using a given observation sequence.

---

## 🧠 Model Information

### Hidden States:
- Rainy (R)
- Sunny (S)

### Observations:
- Walk (W)
- Shop (H)

### Observation Sequence Used:
(W, H, H, W, H)

Numeric representation:
- Walk = 0
- Shop = 1

Sequence:
[0, 1, 1, 0, 1]

---

## ⚙️ Algorithm Implemented

The following steps are implemented in the project:

1. Forward Algorithm (α computation)
2. Backward Algorithm (β computation)
3. Gamma (γ) calculation – state responsibility
4. Xi (ξ) calculation – transition responsibility
5. Parameter Re-estimation:
   - Initial probabilities (π)
   - Transition matrix (A)
   - Emission matrix (B)

The Baum–Welch algorithm runs for multiple iterations until the probabilities converge.

---

## 📂 Project Files

- `hmm.py` → Contains the full implementation of the HMM model and Baum–Welch algorithm.
- `app.py` → Runs the model, trains it, and displays updated probabilities.
- `requirements.txt` → Contains required Python libraries.
- `README.md` → Project documentation.

---

## ▶️ How to Run the Project

### Step 1: Install Python

Make sure Python is installed.

Check using:

```
python --version
```

If not installed, download from:
https://www.python.org/downloads/

---

### Step 2: Download the Project

1. Open the GitHub repository.
2. Click **Code**.
3. Click **Download ZIP**.
4. Extract the folder.

---

### Step 3: Install Required Libraries

Open terminal inside the project folder and run:

```
pip install -r requirements.txt
```

This installs:

- Flask
- NumPy

---

### Step 4: Run the Application

Inside the project folder, run:

```
python app.py
```

---

## 📊 Output

The program displays:

- Initial Transition Matrix
- Initial Emission Matrix
- Initial Initial Probabilities
- Updated Transition Matrix
- Updated Emission Matrix
- Updated Initial Probabilities

These updated values are calculated using the Baum–Welch training process.

---

## 🎯 Learning Outcome

Through this project, the following concepts are understood:

- Hidden Markov Models (HMM)
- Forward and Backward probability computation
- Expectation–Maximization (EM) algorithm
- Baum–Welch parameter estimation
- Probabilistic modeling of sequential data

---

## ✅ Conclusion

This project successfully demonstrates the implementation and training of a Hidden Markov Model using the Baum–Welch algorithm. The model updates its parameters based on the observation sequence provided and outputs the refined probabilities.

---
