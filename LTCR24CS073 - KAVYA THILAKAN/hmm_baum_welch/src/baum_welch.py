import numpy as np


def baum_welch(model, observations, max_iter=100, tol=1e-4):
    """
    Baum-Welch algorithm for training HMM parameters
    Uses log-likelihood for convergence (more stable)
    """
    N = model.N
    M = model.M
    T = len(observations)

    likelihoods = []
    log_likelihoods = []  # Track log-likelihood for convergence

    print(f"Starting Baum-Welch training with {max_iter} max iterations...")

    for iteration in range(max_iter):
        # Forward & Backward
        alpha = model.forward(observations)
        beta = model.backward(observations)

        # Compute likelihood
        likelihood = np.sum(alpha[-1])
        log_likelihood = np.log(likelihood + 1e-300)  # Avoid log(0)

        likelihoods.append(float(likelihood))
        log_likelihoods.append(float(log_likelihood))

        # Convergence check using LOG-LIKELIHOOD (more stable for small probabilities)
        if iteration > 0:
            change = abs(log_likelihoods[-1] - log_likelihoods[-2])
            if change < tol:
                print(f"✓ Converged at iteration {iteration + 1} (change = {change:.6f} < {tol})")
                break

        # Compute gamma (probability of being in state i at time t)
        gamma = np.zeros((T, N), dtype=float)
        for t in range(T):
            denom = np.sum(alpha[t] * beta[t])
            if denom > 0:
                gamma[t] = (alpha[t] * beta[t]) / denom

        # Compute xi (probability of being in state i at time t and state j at time t+1)
        xi = np.zeros((T - 1, N, N), dtype=float)
        for t in range(T - 1):
            denom = np.sum(alpha[t][:, None] * model.A * model.B[:, observations[t + 1]] * beta[t + 1])
            if denom > 0:
                for i in range(N):
                    numer = alpha[t, i] * model.A[i, :] * model.B[:, observations[t + 1]] * beta[t + 1]
                    xi[t, i, :] = numer / denom

        # Update pi
        model.pi = gamma[0].copy()

        # Update A (transition matrix)
        for i in range(N):
            denom = np.sum(gamma[:-1, i])
            if denom > 0:
                for j in range(N):
                    model.A[i, j] = np.sum(xi[:, i, j]) / denom

        # Update B (emission matrix)
        for i in range(N):
            denom = np.sum(gamma[:, i])
            if denom > 0:
                for k in range(M):
                    mask = (np.array(observations) == k)
                    if np.any(mask):
                        model.B[i, k] = np.sum(gamma[mask, i]) / denom
                    else:
                        model.B[i, k] = 1e-8  # small epsilon to avoid zero probabilities

        # Print progress every 5 iterations
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}: Log-Likelihood = {log_likelihood:.6f}")

    print(f"✓ Training completed in {len(likelihoods)} iterations")
    print(f"✓ Final Log-Likelihood: {log_likelihoods[-1]:.6f}")

    return model, likelihoods, log_likelihoods