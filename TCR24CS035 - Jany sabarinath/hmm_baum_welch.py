import numpy as np
import matplotlib.pyplot as plt


class HMM:
    def __init__(self, n_states, n_obs):
        self.n_states = n_states
        self.n_obs = n_obs

        # Random initialization
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(n_states, n_obs)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.n_states))

        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.n_states))

        beta[T-1] = 1

        for t in reversed(range(T-1)):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])

        return beta

    def baum_welch(self, O, n_iter=20):
        T = len(O)
        likelihoods = []

        for _ in range(n_iter):

            alpha = self.forward(O)
            beta = self.backward(O)

            P_O = np.sum(alpha[-1])
            likelihoods.append(P_O)

            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))

            for t in range(T):
                gamma[t] = (alpha[t] * beta[t]) / P_O

            for t in range(T-1):
                denom = np.sum(
                    alpha[t][:, None] * self.A * self.B[:, O[t+1]] * beta[t+1]
                )

                for i in range(self.n_states):
                    numer = alpha[t, i] * self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                    xi[t, i] = numer / denom

            # Update π
            self.pi = gamma[0]

            # Update A
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

            # Update B
            for j in range(self.n_states):
                for k in range(self.n_obs):
                    mask = (np.array(O) == k)
                    self.B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

        return likelihoods


# ---------------- MAIN ----------------

if __name__ == "__main__":

    observed_sequence = [0, 1, 0, 2, 1]
    n_hidden_states = 2
    n_observation_symbols = 3  # Symbols: 0,1,2

    hmm = HMM(n_hidden_states, n_observation_symbols)

    likelihoods = hmm.baum_welch(observed_sequence, n_iter=20)

    print("\nTransition Matrix A:\n", hmm.A)
    print("\nEmission Matrix B:\n", hmm.B)
    print("\nInitial Distribution π:\n", hmm.pi)
    print("\nFinal P(O | λ):", likelihoods[-1])

    plt.plot(likelihoods)
    plt.xlabel("Iteration")
    plt.ylabel("P(O | λ)")
    plt.title("Likelihood vs Iteration")
    plt.show()