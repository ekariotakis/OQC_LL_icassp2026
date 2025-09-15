import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import peeling
import LL_optimization


def densest_onehop_neighborhood(G):
    n = G.number_of_nodes()
    max_density = float('-inf')
    densest_node = None
    densest_neighbors = None
    
    for node in G.nodes():
        # Get 1-hop neighborhood efficiently
        neighbors = set(G.neighbors(node))
        neighborhood_size = len(neighbors) + 1  # Include the node itself
        
        # Count edges in the induced subgraph without creating it
        # Edges = edges among neighbors + edges from node to neighbors
        edges_among_neighbors = 0
        for u in neighbors:
            for v in G.neighbors(u):
                if v in neighbors and u < v:  # Avoid double counting
                    edges_among_neighbors += 1
        
        # Total edges in the induced subgraph
        total_edges = edges_among_neighbors + len(neighbors)
        
        # Compute density - Average degree density = 2 * |E| / |V|
        density = (total_edges) / neighborhood_size
        
        if density > max_density:
            max_density = density
            densest_node = node
            densest_neighbors = neighbors

    print("Densest node:", densest_node)
    print("Max density:", max_density)
    
    # Create indicator vector efficiently
    nodes_list = list(G.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes_list)}
    
    densest_indicator = np.zeros(n, dtype=int)
    densest_indicator[node_to_index[densest_node]] = 1
    for neighbor in densest_neighbors:
        densest_indicator[node_to_index[neighbor]] = 1
    
    print("Indicator vector:\n", densest_indicator)
    
    # Only create the subgraph if actually needed for return
    densest_subgraph = G.subgraph([densest_node] + list(densest_neighbors))
    
    return densest_indicator, densest_subgraph

def compute_smallest_eigenvalue(A):
    eigenvalue, _ = eigsh(A, k=1, which='SA')
    min_eigenvalue_A = eigenvalue[0]
    return min_eigenvalue_A

def compute_largest_eigenvalue(A):
    eigenvalue, _ = eigsh(A, k=1, which='LA')
    max_eigenvalue_A = eigenvalue[0]
    return max_eigenvalue_A

def avg_deg_density(G, weight=None):
    return G.size(weight)/G.number_of_nodes()

def compute_C(min_eigenvalue_A, alpha, n):
    C = -2*min_eigenvalue_A +2*alpha*(n-1)
    return C

def indicator_from_subgraph(G,subgraph):
    # Get a consistent node ordering (important!)
    nodes = list(G.nodes())  # Preserves the order
    node_index = {node: i for i, node in enumerate(nodes)}

    # Create indicator vector
    indicator = np.zeros(len(nodes), dtype=int)
    for node in subgraph:
        indicator[node_index[node]] = 1

    return indicator

def to_indicator_vector(S, n):
    indicator = np.zeros(n, dtype=int)
    for u in S:
        indicator[u] = 1
    return indicator

def indicator_to_set(indicator):
    indicator = np.array(indicator)
    return set(np.where(indicator == 1)[0])

def error_from_binary(x, n):
    return sum(x-x**2)/n