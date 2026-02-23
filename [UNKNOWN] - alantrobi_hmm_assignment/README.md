# HMM Trainer (Baum–Welch Algorithm)

This is a simple Python GUI application that trains a Hidden Markov Model (HMM) using the Baum–Welch algorithm.
The program shows a live log-likelihood graph while training and displays the final transition diagram.

The GUI lets you enter:
- Observation sequence
- Number of hidden states (N)
- Number of symbols (M)
- Number of iterations

Everything works inside a single window.

-------------------------------------------------------------

## 📥 Downloading the Files

1. Go to the GitHub repository.
2. Download BOTH files:
   - BaumWelch.py
   - requirements.txt
3. Save them in the SAME folder on your computer.

Example folder:

HMM_App/
│
├── BaumWelch.py
└── requirements.txt

-------------------------------------------------------------

## 🐍 Python Requirement

Make sure you have Python installed on your system.
Recommended: Python 3.9 or above.

Check your version:

python --version

If Python is not installed, download it from:
https://www.python.org/downloads/

-------------------------------------------------------------

## 📦 Installing Dependencies

Open a terminal (Command Prompt / PowerShell) INSIDE the folder where you downloaded the files.

Run this:

pip install -r requirements.txt

This installs the required packages:
- numpy
- matplotlib
- networkx

-------------------------------------------------------------

## ▶️ Running the Application

Still inside the same folder, run:

python BaumWelch.py

The HMM Trainer window will open.
Enter your inputs and click “Start Training”.

-------------------------------------------------------------

## 📂 Files Included

- BaumWelch.py → Main Python GUI application
- requirements.txt → Required Python packages

-------------------------------------------------------------

## ✔ Notes

- No external software is required
- Works on any system with Python and pip installed
- Keep both files in the same folder when running the program

-------------------------------------------------------------

Enjoy using the HMM Trainer!
