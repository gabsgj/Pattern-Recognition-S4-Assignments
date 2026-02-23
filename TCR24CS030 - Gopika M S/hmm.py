import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = np.array(A)      # Transition matrix
        self.B = np.array(B)      # Emission matrix
        self.pi = np.array(pi)    # Initial probabilities
        self.N = self.A.shape[0]  # Number of states

    # Forward algorithm
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, O[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]
        return alpha

    # Backward algorithm
    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N)
        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])
        return beta

    # Baum-Welch training with log-likelihood tracking
    def baum_welch(self, O, iterations=10):
        T = len(O)
        log_likelihoods = []

        for _ in range(iterations):
            alpha = self.forward(O)
            beta = self.backward(O)

            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            # Compute gamma
            for t in range(T):
                denom = np.sum(alpha[t] * beta[t])
                gamma[t] = (alpha[t] * beta[t]) / denom

            # Compute xi
            for t in range(T-1):
                denom = np.sum(alpha[t] * beta[t])
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (
                            alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]
                        ) / denom

            # Update initial probabilities
            self.pi = gamma[0]

            # Update transition matrix A
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

            # Update emission matrix B
            M = self.B.shape[1]
            for i in range(self.N):
                for k in range(M):
                    mask = (O == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

            # Compute log-likelihood for convergence plot
            ll = np.sum(np.log(np.sum(alpha, axis=1)))
            log_likelihoods.append(ll)

        return log_likelihoods