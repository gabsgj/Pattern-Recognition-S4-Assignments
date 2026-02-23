import matplotlib.pyplot as plt
from graphviz import Digraph
import numpy as np

def get_convergence_plot(log_likelihoods):
    fig = plt.figure(figsize=(10, 6))
    
    if len(log_likelihoods) > 5 and (log_likelihoods[1] - log_likelihoods[0]) > (log_likelihoods[-1] - log_likelihoods[1]) * 10:
        plt.plot(range(1, len(log_likelihoods)), log_likelihoods[1:], marker='o', label="Detail (Iter 1+)")
        plt.title("HMM Convergence: Log-Likelihood vs Iterations (Zoomed)")
    else:
        plt.plot(range(len(log_likelihoods)), log_likelihoods, marker='o')
        plt.title("HMM Convergence: Log-Likelihood vs Iterations")
    
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    return fig

def get_transition_diagram(A, state_names=None):
    n_states = A.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]
    
    dot = Digraph(comment='HMM State Transitions')
    dot.attr(rankdir='LR')
    
    for i in range(n_states):
        dot.node(state_names[i], state_names[i])
        
    for i in range(n_states):
        for j in range(n_states):
            prob = A[i, j]
            if prob > 0.01:
                dot.edge(state_names[i], state_names[j], label=f"{prob:.2f}")
                
    return dot

def get_emissions_plot(B, state_names=None, obs_names=None):
    n_states, n_obs = B.shape
    if state_names is None: state_names = [f"S{i}" for i in range(n_states)]
    if obs_names is None: obs_names = [f"O{i}" for i in range(n_obs)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(B, cmap="YlGnBu")
    
    ax.set_xticks(np.arange(n_obs))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(obs_names)
    ax.set_yticklabels(state_names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(n_states):
        for j in range(n_obs):
            ax.text(j, i, f"{B[i, j]:.2f}", ha="center", va="center", color="black")
            
    ax.set_title("Emission Matrix (B)")
    fig.tight_layout()
    return fig
