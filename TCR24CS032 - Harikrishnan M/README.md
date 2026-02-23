# 🧠 Hidden Markov Model (HMM) Visualizer
> A professional implementation of the Baum-Welch Algorithm with interactive state-transition rendering.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

---

### 👤 Student Information
* **Name:** Harikrishnan M 
* **University Reg No:** TCR24CS032
* **Department:** Computer Science Engineering
* **Project:** Hidden Markov Model Implementation And Visualization 

---
## 📖 Project Overview

This repository features a robust, interactive implementation of a **Hidden Markov Model (HMM)** learner. The primary focus of the project is the **Baum-Welch Algorithm**, a specialized version of the Expectation-Maximization (EM) process used to determine the unknown parameters of a model based on observed data.

While HMMs are often "black boxes," this tool bridges the gap between complex probability theory and visual intuition by providing a real-time, dashboard-driven experience.

### 🧩 Core Functionality
* **Parameter Estimation:** Implements the Forward-Backward procedure to iteratively refine the **Transition Matrix ($A$)** and **Emission Matrix ($B$)**.
* **Dynamic Sequencing:** Users can input custom observation sequences to see how the model adapts its internal state logic to fit the data.
* **High-Fidelity Visualization:** Unlike standard static plots, this project utilizes an advanced rendering engine to generate non-overlapping, professionally styled state-transition diagrams.
* **Convergence Metrics:** Tracks the log-likelihood of the model to ensure the algorithm has reached an optimal local maximum.

### 🔬 Why This Matters
Hidden Markov Models are the backbone of modern technologies like **Speech Recognition**, **Bioinformatics (DNA Sequencing)**, and **Financial Trend Analysis**. This project serves as a practical demonstration of how these models "learn" the hidden structures of the world around us.


### ⚡ Key Features
* **Expectation-Maximization:** Full implementation of the Forward-Backward algorithm.
* **Pro Visualization:** Publication-quality diagrams using the `state-transition-diagrams` engine.
* **Interactive Dashboard:** Sidebar-controlled inputs with real-time metric updates.

---

## 🛠️ Installation & Usage
```bash
git clone [https://github.com/ha7-piixel/Hidden-Markov-Models.git](https://github.com/ha7-piixel/Hidden-Markov-Models.git)
cd Hidden-Markov-Models

pip install -r requirements.txt

streamlit run app.py
