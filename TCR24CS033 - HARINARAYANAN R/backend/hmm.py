import numpy as np

class HMM:
    def __init__(self, num_states, num_observations):
        self.num_states = num_states
        self.num_observations = num_observations
        
        # Initialize randomly with rows summing to 1
        self.A = np.random.dirichlet(np.ones(num_states), size=num_states)
        self.B = np.random.dirichlet(np.ones(num_observations), size=num_states)
        self.pi = np.random.dirichlet(np.ones(num_states))
        
    def get_parameters(self):
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "pi": self.pi.tolist()
        }
