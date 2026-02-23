from state_transition_diagrams import render_state_diagram
import numpy as np
from hmm import HMM
from visualize import plot_likelihood

# -----------------------------
# Example Observation Sequence
# -----------------------------
observations = [0, 1, 0, 1, 1, 0]
n_states = 2
n_observations = 2

# -----------------------------
# Create Model
# -----------------------------
model = HMM(n_states, n_observations)

# -----------------------------
# Train using Baum-Welch
# -----------------------------
likelihoods = model.baum_welch(observations, max_iter=15)

# Clean likelihood printing (remove np.float64)
likelihoods = [float(l) for l in likelihoods]

# -----------------------------
# Print Final Parameters
# -----------------------------
print("Final Transition Matrix A:")
print(model.A)

print("\nFinal Emission Matrix B:")
print(model.B)

print("\nFinal Initial Distribution pi:")
print(model.pi)

print("\nP(O | lambda) per iteration:")
print(likelihoods)

# -----------------------------
# Compute Final Probability
# -----------------------------
_, final_prob = model.forward(observations)
print("\nFinal P(O | lambda):", final_prob)

# -----------------------------
# Plot Likelihood Convergence
# -----------------------------
plot_likelihood(likelihoods)
# -----------------------------
# Generate State Transition Diagram
# -----------------------------
A_matrix = np.array(model.A)

dot = render_state_diagram(
    A_matrix,
    state_labels=[f"S{i}" for i in range(n_states)]
)

dot.render("hmm_state_diagram", format="png", cleanup=True)

print("\nState transition diagram saved as hmm_state_diagram.png")