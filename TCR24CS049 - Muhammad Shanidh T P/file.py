import numpy as np
import matplotlib.pyplot as plt

observations = [0, 1, 1, 0, 1]

# Number of hidden states
N = 2

# Number of observation symbols
M = 2

# Length of observation sequence
T = len(observations)

np.random.seed(42)

pi = np.random.dirichlet(np.ones(N))
A = np.random.dirichlet(np.ones(N), size=N)
B = np.random.dirichlet(np.ones(M), size=N)

iterations = 10
probabilities = []

def forward(pi, A, B, observations):
    alpha = np.zeros((T, N))

    alpha[0] = pi * B[:, observations[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]

    return alpha
def backward(A, B, observations):
    beta = np.zeros((T, N))
    beta[T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, observations[t+1]] * beta[t+1])

    return beta
for iteration in range(iterations):

    print("\n==========================")
    print("Iteration:", iteration+1)
    print("==========================")

    alpha = forward(pi, A, B, observations)
    beta = backward(A, B, observations)

    P_O = np.sum(alpha[T-1])
    probabilities.append(P_O)

    print("\nAlpha:")
    print(alpha)

    print("\nBeta:")
    print(beta)

    gamma = np.zeros((T, N))
    xi = np.zeros((T-1, N, N))

    for t in range(T):
        gamma[t] = (alpha[t] * beta[t]) / P_O

    print("\nGamma:")
    print(gamma)

    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                xi[t, i, j] = (
                    alpha[t, i]
                    * A[i, j]
                    * B[j, observations[t+1]]
                    * beta[t+1, j]
                ) / P_O

    pi = gamma[0]

    for i in range(N):
        for j in range(N):
            A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

    for j in range(N):
        for k in range(M):
            mask = (np.array(observations) == k)
            B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

    print("\nUpdated pi:")
    print(pi)

    print("\nUpdated A:")
    print(A)

    print("\nUpdated B:")
    print(B)

    print("\nProbability P(O|λ):", P_O)
    probabilities.append(P_O)


print("\n==========================")
print("FINAL RESULT")
print("==========================")

print("\nFinal Initial Distribution (pi):")
print(pi)

print("\nFinal Transition Matrix (A):")
print(A)

print("\nFinal Emission Matrix (B):")
print(B)

print("\nFinal Probability P(O|λ):")
print(probabilities[-1])

plt.plot(probabilities, marker='o')
plt.title("Convergence of P(O|λ)")
plt.xlabel("Iteration")
plt.ylabel("Probability")
plt.grid()
plt.savefig("baum_welch_result.png")
