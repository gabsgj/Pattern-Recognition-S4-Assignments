# HMM Baum-Welch Algorithm Implementation

This repository contains an implementation of the Hidden Markov Model (HMM) training using the Baum-Welch algorithm (Expectation-Maximization) from scratch in Python.

## Student Information
- **Name:** Melita Mariam Mathew
- **University Register Number:** TCR24CS046

## Assignment Requirements
- [x] Implement Baum-Welch algorithm.
- [x] Inputs: Observed states sequence and number of hidden states.
- [x] Outputs: Probability of state transitions (A), Emission Matrix (B), Initial Distribution ($\pi$), and $P(O|\lambda)$.
- [x] Visualization: Convergence plot ($P(O|\lambda)$ over iterations) and State Transition Diagram.

## Project Structure
- `hmm.py`: Contains the `HMM` class with Forward, Backward, and Baum-Welch implementations.
- `visualization.py`: Utilities for generating plotting objects for convergence and transition diagrams.
- `app.py`: Interactive Streamlit web-based UI to run the algorithm.

## How to Run

### Prerequisites
- Python 3.8+
- [Optional but recommended] Graphviz installed on your system (for transition diagrams).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/melita-28/hmm-py.git
   cd hmm-baum-welch
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib graphviz
   ```

### Running the Program
Run the following command in your terminal to start the visual interactive web app using Streamlit:
```bash
streamlit run app.py
```
- It will automatically open in a browser where you can enter the Sequence and Number of Hidden States continuously.

## Visualizations
The UI dynamically generates and displays the following plots intuitively in different web tabs:
- **Convergence Plot**: Shows how the log-likelihood $P(O|\lambda)$ changes over iterations.
- **State Transition Diagram**: A directed graph representation showing probabilities between states.
- **Emission Matrix Setup**: A heatmap of the emission matrix.

## Troubleshooting

### Graphviz `ExecutableNotFound` Error
If you see an error saying `failed to execute WindowsPath('dot')`, it means the Graphviz system software is not in your PATH.
1. Download Graphviz from [graphviz.org](https://graphviz.org/download/).
2. During installation, select **"Add Graphviz to the system PATH for all users"**.
3. Restart your terminal/IDE.
