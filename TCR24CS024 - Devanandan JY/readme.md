# HMM Baum-Welch Visualizer 🚀

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://hmm-visualizer.streamlit.app/)

**Built by:** Devanandan JY  
**Register Number:** TCR24CS024  

---

## 📖 Project Overview
An interactive visualization of the Baum-Welch algorithm — the core Expectation-Maximization (EM) method used to train Hidden Markov Models (HMMs). 

The system functions as a visual debugger for the Baum-Welch algorithm, built entirely in Python.

---

## 🎯 Educational Goal

This project is designed to provide an intuitive, visual understanding of complex HMM mechanics:
* Forward-Backward inference
* Soft state assignments ($\gamma$)
* Expected transition counts ($\xi$)
* Expectation-Maximization updates
* Likelihood convergence behavior

---

## ⚙️ What It Does

Set up a custom Hidden Markov Model by defining:
* Hidden states ($N$)
* Emission symbols ($M$)
* Observation sequence

Then observe the algorithm execute step-by-step through:
1. Forward Pass
2. Backward Pass
3. Gamma Computation
4. Xi Computation
5. Parameter Re-estimation ($\pi$, $A$, $B$)

A live log-likelihood chart tracks convergence across iterations, allowing you to visually pinpoint exactly when the model optimizes.

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Hybrid Rendering Engine** | Seamlessly switches between high-speed, flicker-free Matplotlib playback (during animation) and interactive, draggable Pyvis network graphs (when paused). |
| **Real-time Convergence Dashboards** | Live-updating Plotly subplots tracking Log-Likelihood, Observation Probability $P(O\|\lambda)$, Optimization Loss (Negative Log), and Error Rate. |
| **Custom Media Player Dock** | A sticky UI dock for step-by-step algorithm playback with speed controls (1x to 10x). |
| **Dynamic State Diagram** | Animated 3-tier layout (START $\rightarrow$ Hidden States $\rightarrow$ Observations) with dynamic edge widths, "squircle" nodes, and color-coded probability updates. |
| **Dark Mode UI** | Native dark theme optimization for deep visual contrast on active probability wires. |

---

## 💻 Technologies Used

* **Frontend/UI:** Streamlit, Custom CSS Injection
* **Math & Core Logic:** NumPy, Pandas
* **Data Visualization:** Plotly, Pyvis (JS/HTML wrappers), NetworkX, Matplotlib

---

## 🛠️ Run Locally

To run this project locally on your machine, ensure you have Python installed, then follow these steps:

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR-USERNAME/hmm-baum-welch-visualizer.git](https://github.com/YOUR-USERNAME/hmm-baum-welch-visualizer.git)
cd hmm-baum-welch-visualizer

# 2. Install the required dependencies
pip install -r requirements.txt

# 3. Launch the visualizer
streamlit run app.py
