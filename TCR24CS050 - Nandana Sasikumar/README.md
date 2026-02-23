# Baum Welch Algorithm Visualizer

**Name :** Nandana Sasikumar

**University Registration Number :** TCR24CS050

**Course :** Pattern Recognition

**Institution :** Government Engineering College, Thrissur

---

### Project Overview
This project implements the Baum-Welch algorithm, an Expectation-Maximization (EM) technique used to learn the parameters of a Hidden Markov Model (HMM) when only observation sequences are available. The implementation is a web-based application built using Streamlit, allowing for interactive parameter tuning and real-time visualization of model convergence.


### Technical Features
The application performs the following core HMM calculations as defined in the course material:
* **Forward Probability ($\alpha$):** Measures the probability of the prefix evidence.
* **Backward Probability ($\beta$):** Measures the probability of the suffix evidence.
* **Soft Credit Assignment ($\gamma$):** Calculates state responsibility at each time step.
* **Parameter Updates:** Iteratively updates the initial probabilities ($\pi$), transition matrix ($A$), and emission matrix ($B$).

You can access the fully functional web application directly here:  **[Baum_Welch_Algorithm_Visualizer](https://baum-welch-algo-visualizer.streamlit.app/)**

### Local Installation & Execution [Optional]
If you prefer to run the application on your local machine, follow these steps:

#### Prerequisites

Ensure you have Python installed. You will need the following libraries:
* streamlit
* numpy
* matplotlib
* pandas

Follow these steps to set up and run the application on your local machine:

**1. Prepare the Project Directory:**
  * Create a dedicated folder for the project on your computer.
  * Ensure the provided Python script is saved inside this folder and named exactly `app.py`.
    
**2. Install Dependencies:**
  Open your terminal or command prompt and run the following command to install the necessary Python libraries:

  ```bash
  pip install streamlit numpy matplotlib pandas
  ```

**3. Run the Application:**
  * Navigate the terminal to the project directory where your `app.py` file is saved.
  * Execute the following command to launch the web interface:

  ```bash
  streamlit run app.py
  ```
  * The application will automatically open in your default web browser. If it does not, copy the "Local URL" provided in the terminal and paste it into your browser's address bar.

### User Instructions

1. **Configuration:** Use the **Sidebar** to enter your hidden states (e.g., `Sunny Rainy Cloudy`), observation symbols (e.g., `Walk Shop Stay`), and the sequence you wish to train on.
2. **Iterations:** Adjust the slider to set the number of training iterations.
3. **Training:** Click the "Train Model" button.
4. **Analysis:**
    * Expand the **Iteration Tables** to see the $\alpha$, $\beta$, and $\gamma$ values at each step.
    * Review the **Final Parameters** ($A, B, \pi$) in the tables provided.
    * Observe the **Learning Curves** to see the Likelihood $P(O|\lambda)$ increasing and the Uncertainty $1 - P(O|\lambda)$ decreasing.
    
### Visualizations Included
* **$P(O|\lambda)$ Graph:** Shows how the model learns to better explain the data over time.
* **$1 - P(O|\lambda)$ Graph:** Displays the reduction in model uncertainty per iteration.
* **Soft Credit Tables:** Detailed breakdown of state responsibilities per time step.

### Application Preview

#### Parameter Optimization
<img width="1919" height="707" alt="image3" src="https://github.com/user-attachments/assets/37a0f257-a6b6-4f7e-b8ad-1c5fb8a20eae" />

<img width="1919" height="704" alt="image4" src="https://github.com/user-attachments/assets/b0f3f3d1-fb11-4a37-86a8-ff8750d83373" />

#### Learning Curves
<img width="1917" height="713" alt="image5" src="https://github.com/user-attachments/assets/029d2d77-153e-4a53-84c3-3ecaf9c453c9" />


