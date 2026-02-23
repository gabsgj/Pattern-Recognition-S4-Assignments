import numpy as np
import graphviz

class HMM:
    def __init__(self, A, B, pi):
        # Using np.array ensures 'np' is recognized
        self.A = np.array(A, dtype=float)  
        self.B = np.array(B, dtype=float)  
        self.pi = np.array(pi, dtype=float) 

    def _forward(self, obs):
        T = len(obs)
        N = self.A.shape[0]
        alpha = np.zeros((T, N))
        alpha[0] = self.pi * self.B[:, obs[0]]
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = alpha[t-1].dot(self.A[:, j]) * self.B[j, obs[t]]
        return alpha

    def _backward(self, obs):
        T = len(obs)
        N = self.A.shape[0]
        beta = np.zeros((T, N))
        beta[T-1] = np.ones(N)
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = (self.A[i, :] * self.B[:, obs[t+1]]).dot(beta[t+1])
        return beta

    def train(self, obs, n_iter=10):
        T = len(obs)
        N = self.A.shape[0]
        for _ in range(n_iter):
            alpha = self._forward(obs)
            beta = self._backward(obs)
            xi = np.zeros((T-1, N, N))
            for t in range(T-1):
                denom = (alpha[t, :].dot(self.A) * self.B[:, obs[t+1]]).dot(beta[t+1, :])
                for i in range(N):
                    numer = alpha[t, i] * self.A[i, :] * self.B[:, obs[t+1]] * beta[t+1, :]
                    xi[t, i, :] = numer / (denom + 1e-12)

            gamma = np.sum(xi, axis=2)
            gamma = np.vstack((gamma, np.sum(xi[T-2, :, :], axis=0)))
            self.pi = gamma[0] / (np.sum(gamma[0]) + 1e-12)
            self.A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1, 1) + 1e-12)
            for k in range(self.B.shape[1]):
                mask = (obs == k)
                self.B[:, k] = np.sum(gamma[mask, :], axis=0) / (np.sum(gamma, axis=0) + 1e-12)
        return self.A, self.B, self.pi

def draw_hmm(A):
    dot = graphviz.Digraph(comment='HMM States')
    dot.attr(rankdir='LR')
    for i in range(len(A)):
        dot.node(f'S{i}', f'State {i+1}')
    for i in range(len(A)):
        for j in range(len(A[i])):
            prob = round(A[i][j], 2)
            if prob > 0.01:
                dot.edge(f'S{i}', f'S{j}', label=f'{prob}')
    return dot