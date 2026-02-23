import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_log_likelihood(log_likelihoods):
    plt.figure()
    plt.plot(log_likelihoods)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.title("Convergence of Baum-Welch")
    plt.show()


def plot_heatmap(matrix, title):
    plt.figure()
    sns.heatmap(matrix, annot=True, cmap="Blues")
    plt.title(title)
    plt.show()


def plot_transition_diagram(A):
    G = nx.DiGraph()

    n_states = A.shape[0]

    for i in range(n_states):
        G.add_node(f"S{i}")

    for i in range(n_states):
        for j in range(n_states):
            if A[i][j] > 0.01:  # avoid clutter
                G.add_edge(f"S{i}", f"S{j}", weight=round(A[i][j], 2))

    pos = nx.circular_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_size=3000)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("State Transition Diagram")
    plt.show()