import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

class HMMBaumWelch:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        self.A = self.normalize(np.random.rand(self.N, self.N))
        self.B = self.normalize(np.random.rand(self.N, self.M))
        self.pi = self.normalize(np.random.rand(self.N))

    def normalize(self, matrix):
        if matrix.ndim == 1:
            return matrix / np.sum(matrix)
        return matrix / np.sum(matrix, axis=1, keepdims=True)

    # ---------------- FORWARD ----------------
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))

        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    # ---------------- BACKWARD ----------------
    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N)

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])

        return beta

    # ---------------- GAMMA & XI ----------------
    def compute_gamma_xi(self, O, alpha, beta):
        T = len(O)

        gamma = np.zeros((T, self.N))
        xi = np.zeros((T-1, self.N, self.N))

        for t in range(T-1):
            denom = np.sum(alpha[t] * beta[t])

            for i in range(self.N):
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / denom

                for j in range(self.N):
                    xi[t, i, j] = (
                        alpha[t, i]
                        * self.A[i, j]
                        * self.B[j, O[t+1]]
                        * beta[t+1, j]
                    ) / denom

        gamma[T-1] = (alpha[T-1] * beta[T-1]) / np.sum(alpha[T-1] * beta[T-1])

        return gamma, xi

    # ---------------- TRAIN ----------------
    def train(self, O, max_iter=30):
        O = np.array(O)
        likelihoods = []

        for iteration in range(max_iter):
            alpha = self.forward(O)
            beta = self.backward(O)

            gamma, xi = self.compute_gamma_xi(O, alpha, beta)

            # Update pi
            self.pi = gamma[0]

            # Update A
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

            # Update B
            for j in range(self.N):
                for k in range(self.M):
                    mask = (O == k)
                    self.B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

            self.A = self.normalize(self.A)
            self.B = self.normalize(self.B)
            self.pi = self.normalize(self.pi)

            likelihood = np.sum(alpha[-1])
            likelihoods.append(likelihood)

            print(f"Iteration {iteration+1}: P(O|λ) = {likelihood}")

        return likelihoods

    def print_model(self):
        print("\nInitial Distribution (pi):")
        print(self.pi)

        print("\nTransition Matrix (A):")
        print(self.A)

        print("\nEmission Matrix (B):")
        print(self.B)


# ---------------- GRAPH PLOT ----------------
def plot_likelihood(likelihoods):
    plt.plot(likelihoods, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("P(O | λ)")
    plt.title("Likelihood Convergence")
    plt.grid()
    plt.savefig("likelihood_graph.png")
    plt.show()


# ---------------- STATE DIAGRAM ----------------
def draw_state_diagram(A):
    N = A.shape[0]
    dot = Digraph()

    # Create nodes
    for i in range(N):
        dot.node(f"S{i}", f"S{i}")

    # Create edges with probabilities
    for i in range(N):
        for j in range(N):
            prob = round(A[i, j], 3)
            dot.edge(f"S{i}", f"S{j}", label=str(prob))

    dot.render("state_transition_diagram", format="png", cleanup=True)
    print("State diagram saved as state_transition_diagram.png")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Example observation sequence (0,1,2 are symbols)
    O = [0, 1, 2, 1, 0, 2, 1]

    n_states = 2
    n_observations = 3

    hmm = HMMBaumWelch(n_states, n_observations)

    likelihoods = hmm.train(O, max_iter=20)

    hmm.print_model()

    # Plot likelihood graph
    plot_likelihood(likelihoods)

    # Draw state transition diagram
    draw_state_diagram(hmm.A)