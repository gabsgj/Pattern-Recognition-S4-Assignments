import matplotlib.pyplot as plt
import networkx as nx
from hmm import HMM

# -----------------------------
# Train HMM
# -----------------------------
O = [0, 1, 0, 1, 1, 0]
N = 2
M = 2

hmm = HMM(N, M)
hmm.baum_welch(O, iterations=10)
A = hmm.A

# -----------------------------
# Create directed graph
# -----------------------------
G = nx.DiGraph()

for i in range(N):
    G.add_node(f"S{i}")

# Add edges with probabilities
for i in range(N):
    for j in range(N):
        G.add_edge(f"S{i}", f"S{j}", weight=round(A[i][j], 2))

# Fixed positions (important)
pos = {
    "S0": (-1, 0),
    "S1": (1, 0)
}

plt.figure(figsize=(6, 4))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue")

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12)

# Draw curved edges (KEY FIX)
nx.draw_networkx_edges(
    G,
    pos,
    connectionstyle="arc3,rad=0.3",
    arrowsize=20
)

# Edge labels
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("HMM State Transition Diagram")
plt.axis("off")
plt.show()
