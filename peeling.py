import networkx as nx
import random
import utils

def GreedyOQC(G, iterations, alpha=1/3):
    if G.number_of_edges() == 0:
        return 0.0, set()
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    loads = dict.fromkeys(G.nodes, 0)  # Load vector for Greedy++.
    best_density = 0.0  # Highest density encountered.
    best_subgraph = set()  # Nodes of the best subgraph found.

    for _ in range(iterations):
        # Initialize heap for fast access to minimum weighted degree.
        heap = nx.utils.BinaryHeap()

        # Compute initial weighted degrees and add nodes to the heap.
        for node, degree in G.degree:
            heap.insert(node, loads[node] + degree)
        # Set up tracking for current graph state.
        remaining_nodes = set(G.nodes)
        num_edges = G.number_of_edges()
        current_degrees = dict(G.degree)

        while remaining_nodes:
            num_nodes = len(remaining_nodes)

            # Current OQC density of the (implicit) graph
            current_density = compute_degree_function(num_nodes, num_edges, alpha) # <<<
            # current_density = num_edges / num_nodes

            # Update the best density.
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(remaining_nodes)

            # Pop the node with the smallest weighted degree.
            node, _ = heap.pop()
            if node not in remaining_nodes:
                continue  # Skip nodes already removed.

            # Update the load of the popped node.
            loads[node] += current_degrees[node]

            # Update neighbors' degrees and the heap.
            for neighbor in G.neighbors(node):
                if neighbor in remaining_nodes:
                    current_degrees[neighbor] -= 1
                    num_edges -= 1
                    heap.insert(neighbor, loads[neighbor] + current_degrees[neighbor])

            # Remove the node from the remaining nodes.
            remaining_nodes.remove(node)

    return best_density, best_subgraph

def compute_degree_function(num_nodes, num_edges, alpha):
    f_a = num_edges - alpha * num_nodes*(num_nodes-1)/2
    return f_a

def compute_delta(num_nodes, num_edges):
    delta = num_edges / (num_nodes*(num_nodes-1)/2)
    return delta
