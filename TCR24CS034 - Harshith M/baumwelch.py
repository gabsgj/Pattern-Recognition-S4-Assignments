import numpy as np


class HiddenMarkovModel:

    def __init__(self, A, B, pi):

        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.pi = np.array(pi, dtype=float)


    def forward(self, obs):

        T = len(obs)
        N = len(self.A)

        alpha = np.zeros((T, N))

        for i in range(N):
            alpha[0][i] = self.pi[i] * self.B[i][obs[0]]

        for t in range(1, T):
            for j in range(N):

                s = 0
                for i in range(N):
                    s += alpha[t-1][i] * self.A[i][j]

                alpha[t][j] = s * self.B[j][obs[t]]

        return alpha


    def backward(self, obs):

        T = len(obs)
        N = len(self.A)

        beta = np.zeros((T, N))

        for i in range(N):
            beta[T-1][i] = 1

        for t in range(T-2, -1, -1):
            for i in range(N):

                s = 0
                for j in range(N):
                    s += self.A[i][j] * self.B[j][obs[t+1]] * beta[t+1][j]

                beta[t][i] = s

        return beta


    def train(self, obs, iterations=10):

        T = len(obs)
        N = len(self.A)
        M = len(self.B[0])

        for step in range(iterations):

            alpha = self.forward(obs)
            beta = self.backward(obs)

            prob = sum(alpha[T-1])

            gamma = np.zeros((T, N))
            xi = np.zeros((T-1, N, N))

            for t in range(T):
                for i in range(N):
                    gamma[t][i] = (alpha[t][i] * beta[t][i]) / prob


            for t in range(T-1):
                for i in range(N):
                    for j in range(N):

                        xi[t][i][j] = (
                            alpha[t][i]
                            * self.A[i][j]
                            * self.B[j][obs[t+1]]
                            * beta[t+1][j]
                        ) / prob


            for i in range(N):
                self.pi[i] = gamma[0][i]


            for i in range(N):

                denom = sum(gamma[:-1, i])

                for j in range(N):

                    numer = sum(xi[:, i, j])

                    self.A[i][j] = numer / denom


            for i in range(N):

                denom = sum(gamma[:, i])

                for k in range(M):

                    numer = 0

                    for t in range(T):
                        if obs[t] == k:
                            numer += gamma[t][i]

                    self.B[i][k] = numer / denom


        return self.A, self.B, self.pi
