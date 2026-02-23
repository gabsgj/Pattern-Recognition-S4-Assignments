 Hidden Markov Model using Baum–Welch Algorithm

 Student Details
Name: Aneena S S  
University Register Number: TCR24CS012  

 Description
This project implements a Hidden Markov Model (HMM) trained using the Baum–Welch algorithm. The model learns the transition probabilities,
emission probabilities, and initial state distribution from observation sequences.

 Input
- Observation sequence
- Number of hidden states

 Output
- Initial distribution (π)
- Transition matrix (A)
- Emission matrix (B)
- Probability of observation sequence P(O | λ) per iteration

 Algorithm Used
Baum–Welch Algorithm
The Baum–Welch algorithm is an Expectation–Maximization (EM) technique used to train Hidden Markov Models when the hidden states are unknown. It uses:
- Forward algorithm (α)
- Backward algorithm (β)
- State probability (γ)
- Transition probability (ξ)

The parameters π, A, and B are updated iteratively until convergence.

 Visualization
- Line graph showing P(O | λ) vs iterations
- State transition diagram showing learned transition probabilities

 State Transition Diagram
The learned transition probabilities are visualized using a directed graph.Each node represents a hidden state and each edge represents the transition probability between states.

 How to Run
pip install -r requirements.txt
python main.py
python visualize.py
python transition_diagram.py
