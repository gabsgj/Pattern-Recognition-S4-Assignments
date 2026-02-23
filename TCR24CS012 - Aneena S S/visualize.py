import matplotlib.pyplot as plt
from hmm import HMM

O = [0, 1, 0, 1, 1, 0]
N = 2
M = 2
iterations = 15

hmm = HMM(N, M)
likelihoods = hmm.baum_welch(O, iterations)

plt.plot(range(1, iterations+1), likelihoods, marker='o')
plt.xlabel("Iteration")
plt.ylabel("P(O | λ)")
plt.title("Baum-Welch Learning Curve")
plt.grid(True)
plt.show()
