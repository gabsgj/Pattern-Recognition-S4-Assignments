#  Hidden Markov Model Trainer  
### Baum–Welch Algorithm Web Application  

---

##  Student Information  

**Name:** SABEEL T 
**University Register Number:** LTCR24CS076


---

##  Project Overview  

This project is a web-based implementation of a **Hidden Markov Model (HMM)** using the **Baum–Welch algorithm** for parameter learning.

The application is built using:

- Python  
- Streamlit (Web UI Framework)  
- NumPy (Numerical Computation)  
- Matplotlib (Visualization)  
- Graphviz (State Transition Diagram)  

The system allows users to:

- Input an observation sequence  
- Select number of hidden states  
- Train the HMM model  
- Visualize log-likelihood convergence  
- View transition and emission matrices  
- Display state transition graph  

---

##  What is a Hidden Markov Model (HMM)?  

A Hidden Markov Model is a statistical model used to represent systems that:

- Have hidden (unobservable) states  
- Produce observable outputs  
- Follow Markov property (future depends only on present state)  

An HMM consists of:

- **Transition Matrix (A)** → Probability of moving between hidden states  
- **Emission Matrix (B)** → Probability of observation given a state  
- **Initial Distribution (π)** → Starting state probability  

The **Baum–Welch Algorithm** is an Expectation-Maximization (EM) algorithm used to estimate unknown HMM parameters from observation data.

---

##  Features of This Project  

- Interactive web interface  
- Configurable number of hidden states  
- Adjustable training iterations  
- Log-likelihood convergence visualization  
- State transition diagram  
- Matrix display of learned parameters  

---

