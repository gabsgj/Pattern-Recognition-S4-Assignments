import numpy as np


class HMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations

        # Random initialization
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(n_states, n_observations)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()


    # ------------------ SCALED FORWARD ------------------
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)

        # Initialization
        alpha[0] = self.pi * self.B[:, O[0]]
        c[0] = 1.0 / np.sum(alpha[0])
        alpha[0] *= c[0]

        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, O[t]]

            c[t] = 1.0 / np.sum(alpha[t])
            alpha[t] *= c[t]

        log_likelihood = -np.sum(np.log(c))

        return alpha, c, log_likelihood


    # ------------------ SCALED BACKWARD ------------------
    def backward(self, O, c):
        T = len(O)
        beta = np.zeros((T, self.n_states))

        beta[T - 1] = c[T - 1]

        for t in reversed(range(T - 1)):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i] * self.B[:, O[t + 1]] * beta[t + 1]
                )
            beta[t] *= c[t]

        return beta


    # ------------------ BAUM-WELCH ------------------
    def baum_welch(self, O, max_iter=100):
        T = len(O)
        log_likelihoods = []

        for iteration in range(max_iter):

            alpha, c, log_likelihood = self.forward(O)
            beta = self.backward(O, c)

            log_likelihoods.append(log_likelihood)

            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T - 1, self.n_states, self.n_states))

            for t in range(T - 1):
                denom = np.sum(
                    alpha[t][:, None]
                    * self.A
                    * self.B[:, O[t + 1]]
                    * beta[t + 1]
                )

                for i in range(self.n_states):
                    numer = (
                        alpha[t, i]
                        * self.A[i]
                        * self.B[:, O[t + 1]]
                        * beta[t + 1]
                    )
                    xi[t, i] = numer / denom

                gamma[t] = np.sum(xi[t], axis=1)

            gamma[T - 1] = alpha[T - 1] / np.sum(alpha[T - 1])

            # Update pi
            self.pi = gamma[0]

            # Update A
            for i in range(self.n_states):
                self.A[i] = np.sum(xi[:, i, :], axis=0) / np.sum(gamma[:-1, i])

            # Update B
            for i in range(self.n_states):
                for k in range(self.n_observations):
                    mask = (O == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

            # Convergence check
            if iteration > 0:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-4:
                    print(f"Converged at iteration {iteration}")
                    break

        return log_likelihoods