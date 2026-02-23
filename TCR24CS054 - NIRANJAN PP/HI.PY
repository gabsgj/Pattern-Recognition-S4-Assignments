import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    return x / x.sum(axis=1, keepdims=True)

def baum_welch(obs, N, M, iterations=10):
    T = len(obs)

    # Random initialization
    A = normalize(np.random.rand(N, N))
    B = normalize(np.random.rand(N, M))
    pi = np.random.rand(N)
    pi /= pi.sum()

    likelihoods = []

    for it in range(iterations):

        # FORWARD
        alpha = np.zeros((T, N))
        alpha[0] = pi * B[:, obs[0]]

        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]

        # BACKWARD
        beta = np.zeros((T, N))
        beta[-1] = 1

        for t in range(T-2, -1, -1):
            beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])

        # GAMMA
        gamma = (alpha * beta)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # XI
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denom = (alpha[t][:, None] * A * B[:, obs[t+1]] * beta[t+1]).sum()
            xi[t] = (alpha[t][:, None] * A * B[:, obs[t+1]] * beta[t+1]) / denom

        # UPDATE
        pi = gamma[0]
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]

        for k in range(M):
            mask = (obs == k)
            B[:, k] = gamma[mask].sum(axis=0)
        B = normalize(B)

        likelihood = alpha[-1].sum()
        likelihoods.append(likelihood)

    return A, B, pi, likelihoods


# Example
obs = np.array([0,1,0,2,1,0,2,1])
N = 2
M = 3

A, B, pi, likelihoods = baum_welch(obs, N, M)

print("Transition matrix A:\n", A)
print("Emission matrix B:\n", B)
print("Initial pi:\n", pi)
print("P(O|λ):", likelihoods[-1])

plt.plot(likelihoods)
plt.title("Likelihood convergence")
plt.show()
