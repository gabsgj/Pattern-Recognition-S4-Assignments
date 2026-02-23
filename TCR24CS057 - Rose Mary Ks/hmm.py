import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = np.array(A)      # Transition matrix
        self.B = np.array(B)      # Emission matrix
        self.pi = np.array(pi)    # Initial probabilities
        
        self.N = self.A.shape[0]  # Number of states
        self.M = self.B.shape[1]  # Number of observation symbols

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))

        # Initialization
        alpha[0] = self.pi * self.B[:, O[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))

        beta[T-1] = np.ones(self.N)

        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])

        return beta

    def baum_welch(self, O, iterations=10):
        T = len(O)

        for _ in range(iterations):
            alpha = self.forward(O)
            beta = self.backward(O)

            xi = np.zeros((T-1, self.N, self.N))
            gamma = np.zeros((T, self.N))

            for t in range(T-1):
                denom = np.sum(alpha[t] * beta[t])
                for i in range(self.N):
                    numer = alpha[t, i] * self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                    xi[t, i] = numer / denom

            gamma = np.sum(xi, axis=2)
            gamma = np.vstack((gamma, np.sum(xi[T-2], axis=0)))

            # Update pi
            self.pi = gamma[0]

            # Update A
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:, i])

            # Update B
            for i in range(self.N):
                for k in range(self.M):
                    mask = (O == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

        return self.A, self.B, self.pi