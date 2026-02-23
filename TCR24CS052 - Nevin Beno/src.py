import numpy as np
import matplotlib.pyplot as plt
import os


def baum_welch(observations, N, M, max_iter=100):

    T = len(observations)
    np.random.seed(42)

    # ---------------------
    # INITIALIZATION
    # ---------------------
    A = np.random.rand(N, N)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(N, M)
    B /= B.sum(axis=1, keepdims=True)

    pi = np.random.rand(N)
    pi /= pi.sum()

    likelihoods = []

    # =====================================================
    # EM ITERATIONS
    # =====================================================
    for iteration in range(max_iter):

        # ---------------------
        # FORWARD (scaled)
        # ---------------------
        alpha = np.zeros((T, N))
        c = np.zeros(T)

        alpha[0] = pi * B[:, observations[0]]
        c[0] = np.sum(alpha[0]) + 1e-12
        alpha[0] /= c[0]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, observations[t]]
            c[t] = np.sum(alpha[t]) + 1e-12
            alpha[t] /= c[t]

        # ---------------------
        # BACKWARD (scaled)
        # ---------------------
        beta = np.zeros((T, N))
        beta[T - 1] = 1 / c[T - 1]

        for t in reversed(range(T - 1)):
            for i in range(N):
                beta[t, i] = np.sum(
                    A[i] * B[:, observations[t + 1]] * beta[t + 1]
                )
            beta[t] /= c[t]

        # ---------------------
        # GAMMA and XI
        # ---------------------
        gamma = np.zeros((T, N))
        xi = np.zeros((T - 1, N, N))

        for t in range(T - 1):
            denom = np.sum(
                alpha[t][:, None] *
                A *
                B[:, observations[t + 1]] *
                beta[t + 1]
            ) + 1e-12

            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i]

                for j in range(N):
                    xi[t, i, j] = (
                        alpha[t, i]
                        * A[i, j]
                        * B[j, observations[t + 1]]
                        * beta[t + 1, j]
                    ) / denom

        gamma[T - 1] = alpha[T - 1] * beta[T - 1]

        # Normalize gamma rows
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

        # ---------------------
        # RE-ESTIMATION
        # ---------------------

        # Initial probability
        pi = gamma[0]
        pi /= np.sum(pi) + 1e-12

        # Transition matrix
        for i in range(N):
            denom = np.sum(gamma[:-1, i]) + 1e-12
            for j in range(N):
                A[i, j] = np.sum(xi[:, i, j]) / denom

        # Emission matrix
        for i in range(N):
            denom = np.sum(gamma[:, i]) + 1e-12
            for k in range(M):
                mask = (np.array(observations) == k)
                numerator = np.sum(gamma[mask, i])
                B[i, k] = numerator / denom

        # IMPORTANT: force normalization (prevents collapse)
        A /= A.sum(axis=1, keepdims=True) + 1e-12
        B /= B.sum(axis=1, keepdims=True) + 1e-12

        # ---------------------
        # TRUE LOG-LIKELIHOOD
        # ---------------------
        log_likelihood = -np.sum(np.log(c))
        likelihoods.append(log_likelihood)

    return A, B, pi, likelihoods


def visualize_and_save(A, B, likelihoods):

    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    # Likelihood
    plt.figure()
    plt.plot(likelihoods)
    plt.title("Log-Likelihood Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid()
    plt.savefig("screenshots/likelihood.png")
    plt.close()

    # Transition Matrix
    plt.figure()
    plt.imshow(A)
    plt.title("Transition Matrix A")
    plt.colorbar()
    plt.savefig("screenshots/transition_matrix.png")
    plt.close()

    # Emission Matrix
    plt.figure()
    plt.imshow(B)
    plt.title("Emission Matrix B")
    plt.colorbar()
    plt.savefig("screenshots/emission_matrix.png")
    plt.close()


if __name__ == "__main__":

    print("=== Hidden Markov Model Trainer (Baum-Welch) ===")

    N = int(input("Enter number of hidden states: "))
    M = int(input("Enter number of observation symbols: "))

    obs_input = input("Enter observation sequence (space separated integers): ")
    observations = list(map(int, obs_input.strip().split()))

    iterations = int(input("Enter number of training iterations: "))

    A, B, pi, likelihoods = baum_welch(
        observations, N, M, max_iter=iterations
    )

    print("\nTransition Matrix A:\n", A)
    print("\nEmission Matrix B:\n", B)
    print("\nInitial State Probabilities pi:\n", pi)

    visualize_and_save(A, B, likelihoods)

    print("\nPlots saved inside 'screenshots' folder.")