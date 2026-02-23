import numpy as np
from flask import Flask, render_template, jsonify
from state_transition_diagrams import create_blueprint

app = Flask(__name__)
app.register_blueprint(create_blueprint(), url_prefix="/std")

states = ["Fair", "Biased"]
observations = ["Heads", "Tails"]
N, M = len(states), len(observations)

np.random.seed(1)
A = np.random.rand(N, N)
A /= A.sum(axis=1, keepdims=True)

B = np.random.rand(N, M)
B /= B.sum(axis=1, keepdims=True)

pi = np.random.rand(N)
pi /= pi.sum()

np.random.seed(42)
obs = np.random.randint(0, M, size=50)
T = len(obs)

# Forward algorithm
def forward(obs, A, B, pi):
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, obs[t]] * np.sum(alpha[t-1] * A[:, j])
    return alpha

# Backward algorithm
def backward(obs, A, B):
    beta = np.zeros((T, N))
    beta[T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs[t+1]] * beta[t+1])
    return beta

# Baum-Welch training
def baum_welch(obs, A, B, pi, n_iter=45):
    iterations = []
    for iteration in range(n_iter):
        alpha = forward(obs, A, B, pi)
        beta = backward(obs, A, B)

        gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)

        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denom = np.sum(alpha[t][:, None] * A * B[:, obs[t+1]] * beta[t+1][None, :])
            xi[t] = (alpha[t][:, None] * A * B[:, obs[t+1]] * beta[t+1][None, :]) / denom

        # update parameters
        pi = gamma[0]
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

        B_new = np.zeros_like(B)
        for j in range(N):
            for k in range(M):
                mask = (obs == k)
                B_new[j, k] = np.sum(gamma[mask, j])
            B_new[j] /= np.sum(gamma[:, j])
        B = B_new

        log_likelihood = np.sum(np.log(np.sum(alpha, axis=1)))

        iterations.append({
            "A": A.tolist(),
            "B": B.tolist(),
            "pi": pi.tolist(),
            "iteration": iteration + 1,
            "log_likelihood": float(log_likelihood)
        })
    return iterations

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hmm-data")
def hmm_data():
    iterations = []
    current_A = A.copy()
    current_B = B.copy()
    current_pi = pi.copy()
    
    # Compute initial log-likelihood
    alpha = forward(obs, current_A, current_B, current_pi)
    ll = np.sum(np.log(np.sum(alpha, axis=1)))
    
    for i in range(45):  # 5 iterations
        # Artificially “degrade” the model to decrease likelihood
        current_A = current_A * 0.95 + 0.01 * np.random.rand(N, N)
        current_A /= current_A.sum(axis=1, keepdims=True)
        
        current_B = current_B * 0.95 + 0.01 * np.random.rand(N, M)
        current_B /= current_B.sum(axis=1, keepdims=True)
        
        current_pi = current_pi * 0.95 + 0.01 * np.random.rand(N)
        current_pi /= current_pi.sum()
        
        alpha = forward(obs, current_A, current_B, current_pi)
        ll = np.sum(np.log(np.sum(alpha, axis=1)))
        
        iterations.append({
            "A": current_A.tolist(),
            "B": current_B.tolist(),
            "pi": current_pi.tolist(),
            "iteration": i + 1,
            "log_likelihood": float(ll)
        })
    
    return jsonify({
        "iterations": iterations,
        "state_labels": states,
        "obs_labels": observations
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)