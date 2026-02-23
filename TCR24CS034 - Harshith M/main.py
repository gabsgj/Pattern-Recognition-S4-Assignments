from baumwelch import HiddenMarkovModel
from visualize import draw_states


A = [
    [0.7, 0.3],
    [0.4, 0.6]
]

B = [
    [0.1, 0.9],
    [0.6, 0.4]
]

pi = [0.6, 0.4]


observations = [0, 1, 1, 0, 1]


model = HiddenMarkovModel(A, B, pi)

A, B, pi = model.train(observations, 10)


print("Transition Matrix\n")
print(A)

print("\nEmission Matrix\n")
print(B)

print("\nInitial Probabilities\n")
print(pi)


draw_states(A)
