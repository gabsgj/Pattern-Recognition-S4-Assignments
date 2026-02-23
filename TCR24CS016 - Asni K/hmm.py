import numpy as np

class HMM:
    def __init__(self, n_states, n_obs):
        self.N = n_states
        self.M = n_obs

        self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)
        self.pi = np.random.dirichlet(np.ones(self.N))

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

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])

        return beta

    def baum_welch(self, O, max_iters=20):
        T = len(O)
        history = []

        for it in range(max_iters):
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
            gamma = np.vstack((gamma, np.sum(xi[-1], axis=0)))

            self.pi = gamma[0]
            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

            for j in range(self.N):
                for k in range(self.M):
                    mask = (O == k)
                    self.B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

            log_likelihood = np.sum(np.log(np.sum(alpha, axis=1)))
            history.append({
                "A": self.A.tolist(),
                "B": self.B.tolist(),
                "pi": self.pi.tolist(),
                "iteration": it + 1,
                "log_likelihood": float(log_likelihood)
            })

        return history