import numpy as np
import matplotlib.pyplot as plt
from hmm import HiddenMarkovModel
from baum_welch import baum_welch


def main():

    # Example: 2 hidden states, 3 observation symbols
    N = 2
    M = 3

    # Create HMM
    model = HiddenMarkovModel(N, M)

    # Example observation sequence
    observations = [0, 1, 2, 1, 0, 2, 1, 0]

    print("Initial Transition Matrix (A):")
    print(model.A)

    print("\nInitial Emission Matrix (B):")
    print(model.B)

    print("\nInitial Pi:")
    print(model.pi)

    # Train using Baum-Welch
    trained_model, likelihoods = baum_welch(model, observations)

    print("\nTrained Transition Matrix (A):")
    print(trained_model.A)

    print("\nTrained Emission Matrix (B):")
    print(trained_model.B)

    print("\nTrained Pi:")
    print(trained_model.pi)

    print("\nLikelihood values per iteration:")
    print(likelihoods)

    # Plot Likelihood Curve
    plt.plot(likelihoods)
    plt.xlabel("Iteration")
    plt.ylabel("P(O | λ)")
    plt.title("Baum-Welch Convergence")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()