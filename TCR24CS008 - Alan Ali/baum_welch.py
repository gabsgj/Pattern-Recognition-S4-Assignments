import numpy as np

def normalize_rows(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)

def initialize_parameters(N, M):
    A = normalize_rows(np.random.rand(N, N))
    B = normalize_rows(np.random.rand(N, M))
    pi = np.random.rand(N)
    pi /= pi.sum()
    return A, B, pi

def forward(O, A, B, pi):
    T = len(O)
    N = A.shape[0]

    alpha = np.zeros((T, N))
    c = np.zeros(T)

    alpha[0] = pi * B[:, O[0]]
    c[0] = 1.0 / np.sum(alpha[0])
    alpha[0] *= c[0]

    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, O[t]]
        c[t] = 1.0 / np.sum(alpha[t])
        alpha[t] *= c[t]

    log_prob = -np.sum(np.log(c))
    return alpha, c, log_prob

def backward(O, A, B, c):
    T = len(O)
    N = A.shape[0]

    beta = np.zeros((T, N))
    beta[-1] = c[-1]

    for t in reversed(range(T - 1)):
        beta[t] = (A @ (B[:, O[t+1]] * beta[t+1])) * c[t]

    return beta

def baum_welch(O, N, max_iter=100, tol=1e-4):
    M = max(O) + 1
    A, B, pi = initialize_parameters(N, M)

    log_likelihoods = []

    for iteration in range(max_iter):
        alpha, c, log_prob = forward(O, A, B, pi)
        beta = backward(O, A, B, c)

        T = len(O)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T-1, N, N))

        for t in range(T - 1):
            denominator = np.sum(
                alpha[t][:, None] * A * B[:, O[t+1]] * beta[t+1]
            )
            for i in range(N):
                numerator = alpha[t, i] * A[i] * B[:, O[t+1]] * beta[t+1]
                xi[t, i] = numerator / denominator

        pi = gamma[0]

        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]

        for k in range(M):
            mask = (O == k)
            B[:, k] = gamma[mask].sum(axis=0)
        B /= gamma.sum(axis=0)[:, None]

        log_likelihoods.append(log_prob)

        if iteration > 0:
            if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break

    return A, B, pi, log_likelihoods