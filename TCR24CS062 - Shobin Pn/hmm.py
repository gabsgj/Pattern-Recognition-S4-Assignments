import numpy as np

def train_hmm(observations, N, M, max_iter=50):

    T = len(observations)

    np.random.seed(0)

    A = np.random.rand(N, N)
    A = A / A.sum(axis=1, keepdims=True)

    B = np.random.rand(N, M)
    B = B / B.sum(axis=1, keepdims=True)

    pi = np.random.rand(N)
    pi = pi / pi.sum()

    likelihoods = []

    for iteration in range(max_iter):

        alpha = np.zeros((T, N))
        beta = np.zeros((T, N))

        alpha[0] = pi * B[:, observations[0]]

        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * B[:, observations[t]]

        prob = np.sum(alpha[T-1])
        likelihoods.append(prob)

        beta[T-1] = np.ones(N)

        for t in reversed(range(T-1)):
            beta[t] = A @ (B[:, observations[t+1]] * beta[t+1])

        gamma = (alpha * beta) / prob

        xi = np.zeros((T-1, N, N))

        for t in range(T-1):
            denom = np.sum(alpha[t] @ A * B[:, observations[t+1]] * beta[t+1])
            for i in range(N):
                xi[t, i] = alpha[t, i] * A[i] * B[:, observations[t+1]] * beta[t+1] / denom

        pi = gamma[0]
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

        for k in range(M):
            mask = (observations == k)
            B[:, k] = np.sum(gamma[mask], axis=0)

        B = B / np.sum(gamma, axis=0)[:, None]

        if iteration > 0 and abs(likelihoods[-1] - likelihoods[-2]) < 1e-6:
            break

    return A, B, pi, likelihoods,gamma