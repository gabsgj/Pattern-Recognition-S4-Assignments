import numpy as np

class HMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations
        
        # Initialize randomly and normalize
        self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)
        self.pi = np.random.dirichlet(np.ones(self.N))

    def _forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        scaling_factors = np.zeros(T)
        
        alpha[0] = self.pi * self.B[:, obs_seq[0]]
        scaling_factors[0] = 1.0 / np.sum(alpha[0])
        alpha[0] *= scaling_factors[0]
        
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * self.B[j, obs_seq[t]]
            scaling_factors[t] = 1.0 / np.sum(alpha[t])
            alpha[t] *= scaling_factors[t]
        
        return alpha, scaling_factors

    def _backward(self, obs_seq, scaling_factors):
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        
        beta[T-1] = 1.0 * scaling_factors[T-1]
        
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1])
            beta[t] *= scaling_factors[t]
        
        return beta

    def baum_welch(self, obs_seq, n_iter=100):
        T = len(obs_seq)
        log_likelihoods = []

        for iteration in range(n_iter):
            # E-step with scaling to prevent underflow
            alpha, scaling_factors = self._forward(obs_seq)
            beta = self._backward(obs_seq, scaling_factors)
            
            # Log Likelihood calculation from scaling factors
            log_prob_O = -np.sum(np.log(scaling_factors))
            log_likelihoods.append(log_prob_O)
            
            # Gamma
            gamma = (alpha * beta) 
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            
            # Xi
            xi = np.zeros((T - 1, self.N, self.N))
            for t in range(T - 1):
                denominator = np.sum(alpha[t, :] * np.dot(self.A * self.B[:, obs_seq[t+1]], beta[t+1, :])) # This is a bit simplified
                # Re-calculating xi specifically
                for i in range(self.N):
                    numerator = alpha[t, i] * self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1, :]
                    xi[t, i, :] = numerator / np.sum(numerator) if np.sum(numerator) > 0 else 0
                xi[t] *= gamma[t].reshape(-1, 1) # Proper weighting
            
            # Optimized M-step
            self.pi = gamma[0]
            
            # Update A
            for i in range(self.N):
                denom = np.sum(gamma[:-1, i])
                if denom > 0:
                    self.A[i, :] = np.sum(xi[:, i, :], axis=0)
                    self.A[i, :] /= np.sum(self.A[i, :])
                
            # Update B
            for j in range(self.M):
                mask = (obs_seq == j)
                if np.sum(mask) > 0:
                    numerator = np.sum(gamma[mask, :], axis=0)
                    denominator = np.sum(gamma, axis=0)
                    self.B[:, j] = numerator / denominator
            
            # Final normalization check
            self.B /= np.sum(self.B, axis=1, keepdims=True)

        return log_likelihoods

    def get_params(self):
        return {
            "A": self.A,
            "B": self.B,
            "pi": self.pi
        }
