import numpy as np

class HMM:
    def __init__(self, n_states, n_obs):
        self.N = n_states
        self.M = n_obs

        # Random initialization
        self.pi = np.random.rand(self.N)
        self.pi /= np.sum(self.pi)

        self.A = np.random.rand(self.N, self.N)
        self.A /= np.sum(self.A, axis=1, keepdims=True)

        self.B = np.random.rand(self.N, self.M)
        self.B /= np.sum(self.B, axis=1, keepdims=True)

    # ---------- Forward ----------
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))

        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    # ---------- Backward ----------
    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                )

        return beta

    # ---------- Baum-Welch ----------
    def baum_welch(self, O, iterations=10):
        T = len(O)
        likelihoods = []

        for _ in range(iterations):
            alpha = self.forward(O)
            beta = self.backward(O)

            P_O = np.sum(alpha[-1])
            likelihoods.append(P_O)

            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T):
                gamma[t] = (alpha[t] * beta[t]) / P_O

            for t in range(T-1):
                denom = np.sum(
                    alpha[t][:, None] * self.A *
                    self.B[:, O[t+1]] * beta[t+1]
                )
                for i in range(self.N):
                    xi[t, i] = (
                        alpha[t, i] * self.A[i] *
                        self.B[:, O[t+1]] * beta[t+1]
                    ) / denom

            # Update π
            self.pi = gamma[0]

            # Update A
            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

            # Update B
            for k in range(self.M):
                self.B[:, k] = np.sum(gamma[np.array(O) == k], axis=0)

            self.B /= np.sum(gamma, axis=0)[:, None]

        return likelihoods
