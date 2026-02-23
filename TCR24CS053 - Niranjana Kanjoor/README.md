# Hidden Markov Model (HMM) using Baum–Welch Algorithm

## Name
Niranjana Kanjoor

## Register Number
TCR24CS053

## Course
Pattern Recognition  Assignment

---

## Project Description

This project implements a Hidden Markov Model (HMM) using the Baum–Welch algorithm (Expectation-Maximization approach).

The program takes:
- Observation sequence
- Number of hidden states

The program outputs:
- Transition Matrix (A)
- Emission Matrix (B)
- Initial State Distribution (π)
- Probability P(O | λ)
- Log Likelihood convergence graph

The convergence of the log likelihood over iterations is visualized using matplotlib.

---

## Files in Repository

- main.py – Main program file
- hmm.py – HMM class and Baum–Welch implementation
- visualize.py – Function for plotting log likelihood convergence

---

## Requirements

- Python 3.x
- numpy
- matplotlib

Install dependencies using:

pip install numpy matplotlib

---

## How to Run

1. Clone the repository:
git clone <repository-link>

2. Navigate to the project folder:
cd hmm-baum-welch

3. Run the program:
python main.py

4. Enter:
   - Observation sequence (space separated, e.g. 0 1 0 1)
   - Number of hidden states

---

## Output

The program prints:
- Final Log Likelihood
- P(O | λ)
- Transition Matrix (A)
- Emission Matrix (B)
- Initial Distribution (π)

A convergence graph of log likelihood vs iteration is displayed.