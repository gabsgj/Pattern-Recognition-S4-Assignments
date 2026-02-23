from graphviz import Digraph
def create_diagram(A):
    dot = Digraph()
    N = len(A)

    # Add nodes for each state
    for i in range(N):
        dot.node(f"S{i}", f"State {i}")

    # Add edges with transition probabilities
    for i in range(N):
        for j in range(N):
            dot.edge(f"S{i}", f"S{j}", label=str(A[i][j]))

    # Save PNG in static folder
    dot.render("static/state_diagram", format="png", cleanup=True)