import numpy as np

def baum_welch(observations, A, B, pi, max_iter=100, tol=1e-6):

    N = A.shape[0]      # number of states
    M = B.shape[1]      # number of observation symbols
    T = len(observations)

    logs = []

    for iteration in range(max_iter):

        # ---------- FORWARD ----------
        alpha = np.zeros((T, N))

        alpha[0] = pi * B[:, observations[0]]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = B[j, observations[t]] * np.sum(
                    alpha[t - 1] * A[:, j]
                )

        # ---------- BACKWARD ----------
        beta = np.zeros((T, N))
        beta[-1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(
                    A[i] * B[:, observations[t + 1]] * beta[t + 1]
                )

        # ---------- GAMMA & XI ----------
        gamma = np.zeros((T, N))
        xi = np.zeros((T - 1, N, N))

        for t in range(T):
            denom = np.sum(alpha[t] * beta[t])
            gamma[t] = (alpha[t] * beta[t]) / denom

        for t in range(T - 1):
            denom = np.sum(
                alpha[t][:, None] * A * B[:, observations[t + 1]] * beta[t + 1]
            )
            for i in range(N):
                numer = (
                    alpha[t, i]
                    * A[i]
                    * B[:, observations[t + 1]]
                    * beta[t + 1]
                )
                xi[t, i] = numer / denom

        # ---------- UPDATE PARAMETERS ----------
        pi_new = gamma[0]

        A_new = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

        B_new = np.zeros_like(B)

        for j in range(N):
            for k in range(M):
                mask = np.array(observations) == k
                B_new[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

        # ---------- CONVERGENCE CHECK ----------
        diff = max(
            np.max(np.abs(A_new - A)),
            np.max(np.abs(B_new - B)),
            np.max(np.abs(pi_new - pi)),
        )

        logs.append(
            f"Iter {iteration+1}: change = {diff:.6f}"
        )

        A, B, pi = A_new, B_new, pi_new

        if diff < tol:
            logs.append("Converged.")
            break

    return A, B, pi, logs