# 📊 Hidden Markov Model (HMM) Visualizer using Baum-Welch Algorithm

**Name:** SHAZIA  
**Registration Number:** TCR24CS060  
**Course:** B.Tech CSE  
**Assignment:** Implementation of Hidden Markov Model (HMM) using Baum-Welch Algorithm with Visualization  

---

## 📌 Abstract

This project implements a **Hidden Markov Model (HMM)** training system using the **Baum-Welch Algorithm (Expectation-Maximization for HMMs)** and provides an **interactive web-based visualization** of:

- State transition probabilities  
- Emission probabilities  
- Initial state distribution  
- Training convergence (log-likelihood vs iterations)  

The application visually demonstrates how HMM parameters evolve during training and helps users understand probabilistic state transitions through animations and graphs.

---

## 🎯 Objectives

- To implement HMM training using Baum-Welch Algorithm  
- To visualize state transitions and emissions dynamically  
- To plot convergence of log-likelihood during training  
- To build an interactive web application similar to the provided demo  
- To understand probabilistic modeling and EM optimization  

---

## 🧠 Concepts Used

- Hidden Markov Model (HMM)  
- Baum-Welch Algorithm (Forward-Backward + EM)  
- Probability normalization  
- Log-Likelihood maximization  
- Data visualization using D3.js  
- Web application development using Flask  

---

## 🛠️ Tech Stack

**Backend**
- Python  
- Flask  
- NumPy  

**Frontend**
- HTML  
- CSS  
- JavaScript  
- D3.js  

**Visualization Library**
- state-transition-diagrams (for animated HMM diagrams)

---

## ✨ Features

- ✅ Interactive Start → States → Observations Diagram  
- ✅ Animated particle transitions  
- ✅ Inspector panel for A, B, π matrices  
- ✅ Replay controls (play, pause, step)  
- ✅ Log-likelihood convergence graph  
- ✅ Adjustable animation speed  
- ✅ User input for sequences, states, observations  
- ✅ Clean and modern UI  

---

## 📁 Project Structure
hmm-visualizer/
├── app.py
├── requirements.txt
├── templates/
│ └── index.html
└── static/
├── style.css
└── script.js


---



### 1️⃣ Install Python
Download from: https://www.python.org/downloads/

### 2️⃣ Install Required Libraries

```bash
pip install flask numpy state-transition-diagrams
3️⃣ Run the Flask Server
python app.py
4️⃣ Open in Browser
http://localhost:5000
🧪 Sample Input

Sequence

0,1,2,1,0,2,2,1

Hidden States

2
Observations

3
📈 Output

State transition diagram updates at each iteration

Convergence graph shows log-likelihood improving

Model parameters (A, B, π) displayed in inspector panel

Animation shows flow of probability mass