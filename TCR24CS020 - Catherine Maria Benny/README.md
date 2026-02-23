# Hidden Markov Model – Baum–Welch Algorithm Implementation

## Student Details

Name: Catherine Maria Benny 

Register Number: TCR24CS020

Course: Pattern Recognition  

---

## Project Overview

This project implements the Baum–Welch Algorithm, which is the Expectation–Maximization (EM) algorithm used for training Hidden Markov Models (HMMs).

The implementation is written completely from scratch in Python by directly translating the mathematical equations into code.

The project also includes an interactive Streamlit interface to visualize the learning process and the state transition diagram.

---

## Hidden Markov Model Definition

An HMM is defined by the parameter set:

λ = (A, B, π)

Where:

A → State Transition Matrix  
B → Emission Probability Matrix  
π → Initial State Distribution  

The goal of the Baum–Welch algorithm is to estimate these parameters given only an observed sequence.

---

## Algorithm Components Implemented

The following components are implemented in code:

1. Forward Algorithm (α)  
   Computes P(O₁, O₂, ..., Oₜ, qₜ = i | λ)

2. Backward Algorithm (β)  
   Computes P(Oₜ₊₁, ..., Oₜ | qₜ = i, λ)

3. State Responsibility (γ)  
   γₜ(i) = P(qₜ = i | O, λ)

4. Transition Responsibility (ξ)  
   ξₜ(i, j) = probability of transition i → j at time t

5. Parameter Re-Estimation  
   - πᵢ = γ₁(i)  
   - aᵢⱼ = expected transitions i → j / expected transitions from i  
   - bᵢ(o) = expected emissions of symbol o from state i / expected visits to state i  

6. Likelihood Computation  
   P(O | λ) is computed at every iteration to monitor convergence.

---

## Inputs

The application allows configuration of:

- Number of hidden states
- Number of observation symbols
- Observation sequence length
- Maximum number of iterations
- Convergence tolerance

A synthetic observation sequence is generated dynamically for training.

---

## Outputs

After training, the application displays:

- Initial Transition Matrix (A)
- Final Transition Matrix (A)
- Initial Emission Matrix (B)
- Final Emission Matrix (B)
- Initial Distribution (π)
- Final Distribution (π)
- Final Log-Likelihood P(O | λ)

---

## Visualization Features

The application provides:

1. Log-Likelihood convergence plot over iterations  
2. Learned state transition diagram  
3. Transition matrix evolution across iterations  

These visualizations help understand how the parameters improve during training.

---

## Technologies Used

- Python
- NumPy
- Matplotlib
- Streamlit

---


## How to Run the Project

1. Clone the repository

   git clone https://github.com/mariacatheriine/HMM-Baum-Welch-Algo.git
   
   cd HMM-Baum-Welch-Algo  

2. Install dependencies

   pip install -r requirements.txt  

3. Run the application

   streamlit run app.py  

The application will automatically open in your browser.

---

## Notes

- The Baum–Welch algorithm is implemented manually without using any external HMM libraries.
- The implementation follows the mathematical formulation provided in the course material.
- The log-likelihood increases across iterations until convergence.

---

## Conclusion

This project demonstrates a complete implementation of the Baum–Welch algorithm along with visualization tools to understand convergence behavior and learned state transitions in Hidden Markov Models.