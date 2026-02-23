import matplotlib.pyplot as plt
import networkx as nx


def draw_states(A):

    G = nx.DiGraph()

    n = len(A)

    states = []

    for i in range(n):

        state = "State " + str(i)
        states.append(state)

        G.add_node(state)


    for i in range(n):
        for j in range(n):

            prob = round(A[i][j],2)

            G.add_edge(
                states[i],
                states[j],
                weight=prob
            )


    pos = nx.spring_layout(G)

    plt.figure(figsize=(7,6))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        font_size=12
    )

    labels = nx.get_edge_attributes(G,'weight')

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=labels
    )


    plt.title("Hidden Markov Model State Transition Diagram")

    plt.show()
