import numpy as np

def baum_welch(obs_seq, num_states, num_obs, max_iter=50):
    from hmm import HMM
    model = HMM(num_states, num_obs)
    
    A = model.A
    B = model.B
    pi = model.pi
    
    T = len(obs_seq)
    if T == 0:
        return A.tolist(), B.tolist(), pi.tolist(), []
        
    log_likelihoods = []
    
    for iteration in range(max_iter):
        # Forward pass with scaling to prevent underflow
        alpha = np.zeros((T, num_states))
        c = np.zeros(T)
        
        # t=0
        alpha[0] = pi * B[:, obs_seq[0]]
        c[0] = 1.0 / np.sum(alpha[0]) if np.sum(alpha[0]) > 0 else 1.0
        alpha[0] = alpha[0] * c[0]
        
        # t>0
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t-1], A) * B[:, obs_seq[t]]
            c[t] = 1.0 / np.sum(alpha[t]) if np.sum(alpha[t]) > 0 else 1.0
            alpha[t] = alpha[t] * c[t]
            
        # Backward pass
        beta = np.zeros((T, num_states))
        beta[T-1] = c[T-1] # Scaled initialization
        
        for t in range(T-2, -1, -1):
            beta[t] = np.dot(A, (B[:, obs_seq[t+1]] * beta[t+1]))
            beta[t] = beta[t] * c[t]
            
        # Log likelihood
        log_L = -np.sum(np.log(c))
        log_likelihoods.append(log_L)
        
        # E-step
        gamma = np.zeros((T, num_states))
        for t in range(T):
            gamma[t] = (alpha[t] * beta[t]) / c[t] # The c[t] cancels out the double scaling
            gamma_sum = np.sum(gamma[t])
            if gamma_sum > 0:
                gamma[t] /= gamma_sum
                
        xi = np.zeros((T-1, num_states, num_states))
        for t in range(T-1):
            for i in range(num_states):
                for j in range(num_states):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, obs_seq[t+1]] * beta[t+1, j]
            xi_sum = np.sum(xi[t])
            if xi_sum > 0:
                xi[t] /= xi_sum
                
        # M-step
        pi = gamma[0]
        
        gamma_sum_except_last = np.sum(gamma[:-1], axis=0)
        for i in range(num_states):
            if gamma_sum_except_last[i] > 0:
                A[i] = np.sum(xi[:, i, :], axis=0) / gamma_sum_except_last[i]
                
        gamma_sum_all = np.sum(gamma, axis=0)
        for i in range(num_states):
            if gamma_sum_all[i] > 0:
                for k in range(num_obs):
                    # sum over t where O_t = k
                    B[i, k] = np.sum(gamma[np.array(obs_seq) == k, i]) / gamma_sum_all[i]
                    
        # Ensure stochasticity (prevent divide by zero issues)
        A = A / (np.sum(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.sum(B, axis=1, keepdims=True) + 1e-12)
        pi = pi / (np.sum(pi) + 1e-12)
        
    return A.tolist(), B.tolist(), pi.tolist(), log_likelihoods
