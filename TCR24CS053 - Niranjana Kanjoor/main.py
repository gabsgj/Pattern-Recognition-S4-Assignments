import numpy as np
from hmm import HMM
from visualize import plot_log_likelihood

obs_input = input("Enter observation sequence (space separated, e.g. 0 1 0 1): ")
O = np.array([int(x) for x in obs_input.split()])

n_states = int(input("Enter number of hidden states: "))

for states in [2, 3, 4]:
    print(f"\nTraining model with {states} hidden states")

    model = HMM(n_states=states, n_observations=len(set(O)))
    log_likelihoods = model.baum_welch(O, max_iter=50)

    print("\nFinal Results")

    final_log_likelihood = log_likelihoods[-1]
    print("Final Log Likelihood:", final_log_likelihood)

    prob = np.exp(final_log_likelihood)
    print("P(O | lambda):", prob)

    print("\nTransition Matrix (A):\n", model.A)
    print("\nEmission Matrix (B):\n", model.B)
    print("\nInitial Distribution (pi):\n", model.pi)

    plot_log_likelihood(log_likelihoods)