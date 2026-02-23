import numpy as np

class HMM:
    def __init__(self, n_states, n_obs):
        self.n_states = n_states
        self.n_obs = n_obs
        
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
                alpha[t, j] = self.B[j, O[t]] * np.sum(alpha[t-1] * self.A[:, j])
        
        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.ones((T, self.n_states))
        
        for t in reversed(range(T-1)):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])
        
        return beta

    def baum_welch(self, O, n_iter=10):
        T = len(O)

        for _ in range(n_iter):
            alpha = self.forward(O)
            beta = self.backward(O)

            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                denom = np.sum(alpha[t] * beta[t])
                for i in range(self.n_states):
                    numer = alpha[t, i] * self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                    xi[t, i] = numer / denom

            gamma = np.sum(xi, axis=2)

            self.pi = gamma[0]
            self.A = np.sum(xi, axis=0) / np.sum(gamma, axis=0, keepdims=True).T
            
            gamma = np.vstack((gamma, np.sum(xi[T-2], axis=0)))
            for j in range(self.n_states):
                for k in range(self.n_obs):
                    mask = (O == k)
                    self.B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])