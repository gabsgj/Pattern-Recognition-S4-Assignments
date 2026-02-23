# Hidden Markov Model (Baum-Welch) Visual Trainer

## Student Details
Name: Madhav Krishna T  
Register Number: TCR24CS043

---

## Live Application

The deployed web application can be accessed here:

https://hmm-baum-welch-implementation.onrender.com

---

## Project Overview

This project implements a Hidden Markov Model (HMM) trained using the Baum-Welch algorithm (Expectation-Maximization method).  
The system learns model parameters from an observed sequence and provides interactive visualizations of the training process.

The application runs as a web dashboard built with:

• Python (Flask backend)  
• NumPy (numerical computation)  
• HTML, CSS, Bootstrap (UI)  
• Chart.js (visualizations)  

---

## Features

### Training Module
• Accepts observed sequence as input  
• Accepts number of hidden states  
• Accepts number of observation symbols  
• Runs Baum-Welch iterations  
• Computes convergence automatically  

### Outputs Generated
• Transition matrix A  
• Emission matrix B  
• Initial distribution π  
• Probability of observation sequence P(O|λ)  
• Log-Likelihood at each iteration  
• Negative Log-Likelihood  
• Convergence statistics  

### Visualizations
• Log-Likelihood convergence graph  
• Observation probability graph  
• Optimization loss graph  
• Probability complement graph  
• State transition diagram  
• Heatmap matrices for A, B, and π  

---

## Project Structure

project/
│
├── app.py                → Flask backend with Baum-Welch implementation  
├── templates/index.html  → Dashboard UI  
├── requirements.txt      → Dependencies  
├── Procfile              → Deployment config  
└── README.md  

---

## How to Run Locally

1. Install dependencies

pip install flask numpy

2. Run the server

python app.py

3. Open browser and go to

http://127.0.0.1:5000

---

## How to Use

1. Enter an observed sequence  
   Example:  
   0 1 2 1 0 2 1 2  

2. Enter:
   • Number of hidden states  
   • Number of observation symbols  
   • Number of iterations  

3. Click **Train Model**

4. The dashboard will display:
   • Convergence graphs  
   • State transition diagram  
   • Learned matrices A, B, π  
   • Iteration log  

---

## Algorithm Description

This project uses the Baum-Welch algorithm, which is a special case of the Expectation-Maximization (EM) algorithm.

### Steps of the Algorithm

1. Initialize matrices A, B and vector π randomly  
2. Compute Forward probabilities α  
3. Compute Backward probabilities β  
4. Compute state probabilities γ  
5. Compute transition probabilities ξ  
6. Update A, B, and π using expectations  
7. Repeat until convergence  

---

## Educational Purpose

This project demonstrates:

• Hidden state learning from data  
• Expectation-Maximization optimization  
• Convergence behaviour of probabilistic models  
• Visualization of parameter evolution  

---

## Status

✔ Baum-Welch implementation complete  
✔ Visualization dashboard complete  
✔ Interactive training system complete  
✔ Live deployment available  
✔ Ready for submission  
