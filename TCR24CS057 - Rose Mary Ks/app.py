from flask import Flask, render_template_string
import numpy as np
from hmm import HMM

app = Flask(__name__)

@app.route("/")
def home():
    # Rainy (0), Sunny (1)
    A = [[0.7, 0.3],
         [0.4, 0.6]]

    B = [[0.1, 0.9],   # Rainy -> Walk, Shop
         [0.6, 0.4]]   # Sunny -> Walk, Shop

    pi = [0.6, 0.4]

    observations = np.array([0,1,1,0,1])  # W,H,H,W,H

    model = HMM(A, B, pi)
    A_new, B_new, pi_new = model.baum_welch(observations, iterations=5)

    return f"""
    <h1>HMM Baum-Welch Training</h1>
    <h2>Updated Transition Matrix:</h2>
    {A_new}
    <h2>Updated Emission Matrix:</h2>
    {B_new}
    <h2>Updated Initial Probabilities:</h2>
    {pi_new}
    <br><br>
    <img src="https://i.imgur.com/8G3J6Qx.png" width="400">
    """

if __name__ == "__main__":
    app.run(debug=True)