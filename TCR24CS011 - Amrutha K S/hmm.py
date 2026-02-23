import numpy as np

class HMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations

        # Random Initialization
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(n_states, n_observations)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()

    # ------------------ FORWARD ------------------
    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialization
        alpha[0] = self.pi * self.B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]

        probability = np.sum(alpha[T-1])
        return alpha, probability

    # ------------------ BACKWARD ------------------
    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialization
        beta[T-1] = np.ones(self.n_states)

        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i] *
                    self.B[:, observations[t+1]] *
                    beta[t+1]
                )

        return beta

    # ------------------ GAMMA & XI ------------------
    def compute_gamma_xi(self, observations, alpha, beta):
        T = len(observations)
        gamma = np.zeros((T, self.n_states))
        xi = np.zeros((T-1, self.n_states, self.n_states))

        for t in range(T-1):
            denom = np.sum(alpha[t] * beta[t])

            for i in range(self.n_states):
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / denom

                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.A[i, j] *
                        self.B[j, observations[t+1]] *
                        beta[t+1, j]
                    ) / denom

        gamma[T-1] = (alpha[T-1] * beta[T-1]) / np.sum(alpha[T-1] * beta[T-1])

        return gamma, xi

    # ------------------ BAUM-WELCH ------------------
    def baum_welch(self, observations, max_iter=20):
        likelihoods = []

        for iteration in range(max_iter):

            alpha, prob = self.forward(observations)
            beta = self.backward(observations)

            gamma, xi = self.compute_gamma_xi(observations, alpha, beta)

            likelihoods.append(prob)

            # Update pi
            self.pi = gamma[0]

            # Update A
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

            # Update B
            for j in range(self.n_states):
                for k in range(self.n_observations):
                    mask = np.array(observations) == k
                    self.B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

        return likelihoods