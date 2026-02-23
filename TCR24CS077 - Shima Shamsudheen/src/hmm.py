import numpy as np

class HiddenMarkovModel:
    def __init__(self, N, M):
        """
        N = number of hidden states
        M = number of observation symbols
        """
        self.N = N
        self.M = M
        
        # Random initialization
        self.A = np.random.rand(N, N)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        self.B = np.random.rand(N, M)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)
        
        self.pi = np.random.rand(N)
        self.pi = self.pi / self.pi.sum()
    
    def forward(self, observations):
        """Forward algorithm"""
        T = len(observations)
        alpha = np.zeros((T, self.N))
        
        alpha[0] = self.pi * self.B[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]
        
        return alpha
    
    def backward(self, observations):
        """Backward algorithm"""
        T = len(observations)
        beta = np.zeros((T, self.N))
        
        beta[T-1] = 1
        
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1])
        
        return beta
    
    def viterbi(self, observations):
        """Viterbi algorithm"""
        T = len(observations)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        
        delta[0] = self.pi * self.B[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.N):
                temp = delta[t-1] * self.A[:, j] * self.B[j, observations[t]]
                delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(temp)
        
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states