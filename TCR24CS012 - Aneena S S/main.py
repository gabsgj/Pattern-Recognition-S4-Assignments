from hmm import HMM

# INPUTS
O = list(map(int, input("Enter observation sequence: ").split()))
N = int(input("Enter number of hidden states: "))
M = len(set(O))
iterations = 10

# MODEL
hmm = HMM(N, M)
likelihoods = hmm.baum_welch(O, iterations)

# OUTPUTS
print("\nInitial Distribution (π):")
print(hmm.pi)

print("\nTransition Matrix (A):")
print(hmm.A)

print("\nEmission Matrix (B):")
print(hmm.B)

print("\nP(O | λ) per iteration:")
for i, val in enumerate(likelihoods):
    print(f"Iteration {i+1}: {val}")
