# Hidden Markov Model - Baum Welch Algorithm

Name: Nandana Mk  
Register Number: TCR24CS047 

## Description
This project implements the Hidden Markov Model (HMM) training using the Baum-Welch algorithm.

Inputs:
- Observed state sequence
- Number of hidden states

Outputs:
- Transition Matrix A
- Emission Matrix B
- Initial Distribution pi
- Log Likelihood P(O | λ)
- State transition diagram
- Convergence graph

## Requirements
Python 3.x  
numpy  
matplotlib  
networkx  

Install dependencies:

pip install numpy matplotlib networkx

## How to Run

python hmm_baum_welch.py

## Features
- Forward Algorithm
- Backward Algorithm
- Gamma and Xi computation
- Parameter re-estimation
- Convergence visualization
- State transition diagram