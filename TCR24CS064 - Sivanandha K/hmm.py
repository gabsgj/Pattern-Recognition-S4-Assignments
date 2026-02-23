import numpy as np

def forward(O, A, B, pi):
    T = len(O)
    N = len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, O[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, O[t]]
    return alpha


def backward(O, A, B):
    T = len(O)
    N = A.shape[0]
    beta = np.zeros((T, N))
    beta[T-1] = 1

    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, O[t+1]] * beta[t+1])
    return beta


def baum_welch(O, N, M, max_iter=20):

    T = len(O)

    # Random initialization
    pi = np.random.rand(N)
    pi /= pi.sum()

    A = np.random.rand(N, N)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(N, M)
    B /= B.sum(axis=1, keepdims=True)

    log_likelihoods = []

    for _ in range(max_iter):

        alpha = forward(O, A, B, pi)
        beta = backward(O, A, B)

        P = np.sum(alpha[-1])
        log_likelihoods.append(np.log(P))

        gamma = (alpha * beta) / P

        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denom = np.sum(alpha[t][:, None] * A * B[:, O[t+1]] * beta[t+1])
            for i in range(N):
                numer = alpha[t, i] * A[i] * B[:, O[t+1]] * beta[t+1]
                xi[t, i] = numer / denom

        # Update pi
        pi = gamma[0]

        # Update A
        for i in range(N):
            A[i] = np.sum(xi[:, i, :], axis=0) / np.sum(gamma[:-1, i])

        # Update B
        for i in range(N):
            for k in range(M):
                mask = (O == k)
                B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

    return pi, A, B, P, log_likelihoods
