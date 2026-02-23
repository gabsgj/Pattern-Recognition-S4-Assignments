"""
Hidden Markov Model using Baum-Welch Algorithm
Author: Archith Sunil
Register Number: TCR24CSXXXX
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ===============================
# Forward Algorithm
# ===============================
def forward(pi, A, B, O):
    T = len(O)
    N = len(pi)
    alpha = np.zeros((T, N))

    # Initialization
    alpha[0] = pi * B[:, O[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, O[t]]

    return alpha


# ===============================
# Backward Algorithm
# ===============================
def backward(A, B, O):
    T = len(O)
    N = A.shape[0]
    beta = np.zeros((T, N))

    # Initialization
    beta[T - 1] = 1

    # Recursion
    for t in reversed(range(T - 1)):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, O[t + 1]] * beta[t + 1])

    return beta


# ===============================
# Plot Likelihood Graph
# ===============================
def plot_likelihood(likelihoods):
    plt.figure(figsize=(8, 5))
    plt.plot(likelihoods, marker='o')
    plt.title("Likelihood vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("P(O | λ)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("likelihood_plot.png", dpi=300)
    plt.show()


# ===============================
# Draw State Transition Diagram
# ===============================
def draw_transition_diagram(A):
    G = nx.DiGraph()
    N = A.shape[0]

    for i in range(N):
        G.add_node(f"S{i}")

    for i in range(N):
        for j in range(N):
            G.add_edge(f"S{i}", f"S{j}", weight=round(A[i][j], 3))

    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True,
            node_size=3000,
            node_color='lightblue')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("State Transition Diagram")
    plt.savefig("state_transition.png", dpi=300)
    plt.show()


# ===============================
# MAIN PROGRAM
# ===============================
if __name__ == "__main__":

    # Input
    obs_input = input("Enter observation sequence (space separated numbers, e.g., 0 1 0): ")
    O = np.array(list(map(int, obs_input.split())))

    N = int(input("Enter number of hidden states: "))
    M = len(np.unique(O))
    T = len(O)

    # Random Initialization
    np.random.seed(42)

    pi = np.random.rand(N)
    pi /= np.sum(pi)

    A = np.random.rand(N, N)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(N, M)
    B /= B.sum(axis=1, keepdims=True)

    likelihoods = []

    max_iterations = 100
    tolerance = 1e-6
    prev_likelihood = 0

    # ===============================
    # Baum-Welch Training
    # ===============================
    for iteration in range(max_iterations):

        alpha = forward(pi, A, B, O)
        beta = backward(A, B, O)

        gamma = np.zeros((T, N))
        xi = np.zeros((T - 1, N, N))

        for t in range(T - 1):
            denom = np.sum(alpha[t] * beta[t])
            for i in range(N):
                gamma[t][i] = (alpha[t][i] * beta[t][i]) / denom
                for j in range(N):
                    xi[t][i][j] = (
                        alpha[t][i]
                        * A[i][j]
                        * B[j][O[t + 1]]
                        * beta[t + 1][j]
                    ) / denom

        gamma[T - 1] = alpha[T - 1] / np.sum(alpha[T - 1])

        # Re-estimate pi
        pi = gamma[0]

        # Re-estimate A
        for i in range(N):
            for j in range(N):
                A[i][j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

        # Re-estimate B
        for i in range(N):
            for k in range(M):
                mask = (O == k)
                B[i][k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

        likelihood = np.sum(alpha[-1])
        likelihoods.append(likelihood)

        if abs(likelihood - prev_likelihood) < tolerance:
            print("Model Converged at iteration:", iteration)
            break

        prev_likelihood = likelihood

    # ===============================
    # OUTPUT
    # ===============================
    print("\nFinal Initial Distribution (pi):\n", pi)
    print("\nFinal Transition Matrix (A):\n", A)
    print("\nFinal Emission Matrix (B):\n", B)
    print("\nFinal Likelihood P(O|lambda):\n", likelihood)

    # Visualization
    plot_likelihood(likelihoods)
    draw_transition_diagram(A)

    print("\nProgram Finished Successfully.")