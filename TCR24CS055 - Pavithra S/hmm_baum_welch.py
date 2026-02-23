import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.pi = np.array(pi, dtype=float)
        self.N = self.A.shape[0]

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1

        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i] *
                    self.B[:, O[t+1]] *
                    beta[t+1]
                )
        return beta

    def baum_welch(self, O, iterations=10):
        T = len(O)
        log_likelihoods = []
        A_history = []

        for _ in range(iterations):

            alpha = self.forward(O)
            beta = self.backward(O)

            P_O = np.sum(alpha[-1])
            log_likelihoods.append(np.log(P_O + 1e-10))

            gamma = (alpha * beta) / (P_O + 1e-10)
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T-1):
                denom = np.sum(
                    alpha[t][:, None] *
                    self.A *
                    self.B[:, O[t+1]] *
                    beta[t+1]
                )

                for i in range(self.N):
                    numer = (
                        alpha[t, i] *
                        self.A[i] *
                        self.B[:, O[t+1]] *
                        beta[t+1]
                    )
                    xi[t, i] = numer / (denom + 1e-10)

            self.pi = gamma[0]

            for i in range(self.N):
                self.A[i] = np.sum(xi[:, i, :], axis=0) / (
                    np.sum(gamma[:-1, i]) + 1e-10
                )

            M = self.B.shape[1]
            for i in range(self.N):
                for k in range(M):
                    mask = (O == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / (
                        np.sum(gamma[:, i]) + 1e-10
                    )

            A_history.append(self.A.copy())

        return self.A, self.B, self.pi, log_likelihoods, A_history