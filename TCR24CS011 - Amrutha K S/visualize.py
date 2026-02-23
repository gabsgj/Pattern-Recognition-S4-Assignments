import matplotlib.pyplot as plt

def plot_likelihood(likelihoods):
    plt.plot(likelihoods)
    plt.xlabel("Iterations")
    plt.ylabel("P(O | λ)")
    plt.title("Baum-Welch Convergence")
    plt.show()