# Hidden Markov Model using Baum-Welch Algorithm

## Student Details
Name: Archith Sunil  
Register Number: TCR24CS015  

---

## Project Description

This project implements a Hidden Markov Model (HMM) trained using the Baum-Welch Algorithm (Expectation-Maximization method).

The program:

• Accepts an observation sequence  
• Randomly initializes HMM parameters (π, A, B)  
• Applies Forward and Backward algorithms  
• Computes γ (gamma) and ξ (xi)  
• Re-estimates model parameters iteratively  
• Stops when convergence condition is satisfied  
• Plots Likelihood vs Iterations graph  
• Generates State Transition Diagram  

---

## Mathematical Concepts Used

1. Forward Algorithm (α computation)
2. Backward Algorithm (β computation)
3. Expectation Step (γ and ξ calculation)
4. Maximization Step (Re-estimation of π, A, B)
5. Convergence checking using tolerance

---

## Technologies Used

- Python
- NumPy
- Matplotlib
- NetworkX

---

## How to Run the Program

### Step 1: Install Dependencies

pip install numpy matplotlib networkx

### Step 2: Run the Program

python main.py

### Step 3: Provide Inputs

Example:

Enter observation sequence:  
0 1 0  

Enter number of hidden states:  
2  

---

## Output

The program outputs:

• Final Initial Distribution (π)  
• Final Transition Matrix (A)  
• Final Emission Matrix (B)  
• Final Likelihood P(O|λ)  
• Likelihood graph saved as `likelihood_plot.png`  
• State transition diagram saved as `state_transition.png`  

---

## Repository Structure

HMM_BaumWelch/
│
├── main.py  
├── README.md  
├── requirements.txt  
├── likelihood_plot.png  
├── state_transition.png  

---

## Conclusion

This project demonstrates the complete implementation of the Baum-Welch training algorithm for Hidden Markov Models along with visual representation of training convergence and state transitions.