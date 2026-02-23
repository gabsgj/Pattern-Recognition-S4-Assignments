# Pattern Recognition Project: Baum-Welch Algorithm Visualizer

This project is a Streamlit application that provides a step-by-step visualization of the Baum-Welch algorithm, a special case of the Expectation-Maximization algorithm used to find the unknown parameters of a Hidden Markov Model (HMM).

## Student Information

- **Name:** Sooraj K R
- **University Registration Number:** TCR24CS065

## Description

The application interactively demonstrates the core calculations of the Baum-Welch algorithm:

- **Forward Pass (Alpha):** Calculates the probability of seeing the observation sequence up to a certain time, given the state.
- **Backward Pass (Beta):** Calculates the probability of the rest of the observation sequence, given the state at a certain time.
- **Gamma:** The probability of being in a particular state at a particular time.
- **Xi:** The probability of transitioning from one state to another.
- **Parameter Updates:** Shows how the initial probabilities (pi), transition matrix (A), and emission matrix (B) are updated based on the calculations.
- **Full Iteration:** Demonstrates multiple iterations of the algorithm to show how the parameters converge.

## How to Run the Code

1.  **Prerequisites:**
    *   Python installed on your system.
    *   `streamlit`, `numpy`, and `pandas` libraries installed. You can install them using pip:
        ```bash
        pip install streamlit numpy pandas
        ```
2.  **Run from VS Code:**
    *   Open the project folder in Visual Studio Code.
    *   Open the `app.py` file.
    *   Open a new terminal in VS Code.
    *   Run the following command in the terminal:
        ```bash
        streamlit run app.py
        ```
3.  The application will open in your web browser.

## Screenshots

Here are some screenshots of the application:

![Screenshot 1](images/Screenshot%202026-02-21%20130434.png)
![Screenshot 2](images/Screenshot%202026-02-21%20130444.png)

