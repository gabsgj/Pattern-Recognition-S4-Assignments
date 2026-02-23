import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



class HMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        # Initialize Transition Matrix A
        self.A = np.random.rand(self.N, self.N)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Initialize Emission Matrix B
        self.B = np.random.rand(self.N, self.M)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

        # Initialize Initial State Distribution pi
        self.pi = np.random.rand(self.N)
        self.pi = self.pi / self.pi.sum()
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))

        # 1️⃣ Initialization
        alpha[0] = self.pi * self.B[:, O[0]]

        # 2️⃣ Recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha
    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))

        # 1️⃣ Initialization
        beta[T-1] = 1

        # 2️⃣ Recursion
        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                )

        return beta
    def baum_welch(self, O, max_iter=10):
        T = len(O)
        log_likelihoods = []

        for iteration in range(max_iter):

            alpha = self.forward(O)
            beta = self.backward(O)

            # Compute P(O | lambda)
            P_O = np.sum(alpha[-1])
            log_likelihoods.append(np.log(P_O))

            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            # Compute gamma
            for t in range(T):
                denominator = np.sum(alpha[t] * beta[t])
                gamma[t] = (alpha[t] * beta[t]) / denominator

            # Compute xi
            for t in range(T-1):
                denominator = np.sum(
                    alpha[t][:, None] *
                    self.A *
                    self.B[:, O[t+1]] *
                    beta[t+1]
                )

                for i in range(self.N):
                    numerator = (
                        alpha[t, i] *
                        self.A[i] *
                        self.B[:, O[t+1]] *
                        beta[t+1]
                    )
                    xi[t, i] = numerator / denominator

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

        return log_likelihoods


def draw_hmm(A):
    G = nx.DiGraph()

    for i in range(len(A)):
        for j in range(len(A)):
            G.add_edge(f"S{i}", f"S{j}", weight=round(A[i][j], 2))

    pos = nx.circular_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


if __name__ == "__main__":
    O = np.array([0, 1, 0, 1, 1, 0, 1, 0])

    model = HMM(n_states=2, n_observations=2)

    log_likelihoods = model.baum_welch(O, max_iter=10)

    print("Final Transition Matrix A:")
    print(model.A)

    print("\nFinal Emission Matrix B:")
    print(model.B)

    print("\nFinal Initial Distribution pi:")
    print(model.pi)

    print("\nLog Likelihood per iteration:")
    print(log_likelihoods)

    plt.plot(log_likelihoods)
    plt.xlabel("Iteration")
    plt.ylabel("Log P(O|λ)")
    plt.title("Baum-Welch Convergence")
    plt.show()


#state-transistion
    draw_hmm(model.A)