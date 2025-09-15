import os
import copy
import numpy as np
import networkx as nx

from datetime import datetime
import matplotlib.pyplot as plt 
from tqdm import tqdm

import scipy
from scipy.sparse.linalg import eigsh

from init_graph import *
import peeling
import utils
import LL_optimization

import argparse

def main():

    logs_save_path = "/logs/"

    parser = argparse.ArgumentParser(description='Run NX PR Iters')
    parser.add_argument('--dataset-name', type=str, default='facebook_combined', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--alpha', type=float, default=1/3, metavar='R', help='Alpha Value')
    # parser.add_argument('--alpha-onehop', type=float, default=0.9, metavar='R', help='Alpha Value for One-Hop Neighborhood')
    parser.add_argument('--initialization', type=str, default='S_oh', metavar='S', help='Initialization')
    parser.add_argument('--tolerance', type=float, default=1e-1, metavar='R', help='Tolerance')
    parser.add_argument('--lam', type=float, default=1e5, metavar='R', help='lambda')
    parser.add_argument('--mu', type=float, default=1e5/2, metavar='R', help='mu')
    parser.add_argument('--dec-rate', type=float, default=0.9, metavar='R', help='Decrease Rate')
    parser.add_argument('--method', type=str, default='L-BFGS-B', metavar='S', help='Optimization Method')
    parser.add_argument('--max-iters', type=int, default=20, metavar='R', help='Maximum Number of Iterations')
    
    args = parser.parse_args()
    dataset_name = copy.deepcopy(args.dataset_name)
    alpha = copy.deepcopy(args.alpha)
    # alpha_onehop = copy.deepcopy(args.alpha_onehop)
    initialization = copy.deepcopy(args.initialization)
    tolerance = copy.deepcopy(args.tolerance)
    lam = copy.deepcopy(args.lam)
    mu = copy.deepcopy(args.mu)
    dec_rate = copy.deepcopy(args.dec_rate)
    method = copy.deepcopy(args.method)
    max_iters = copy.deepcopy(args.max_iters)

    print()
    print("------------------------------")
    print("Dataset Name:", dataset_name)

    source_path = "../datasets/"
    G = init_graph(dataset_name, source_path)

    print(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    A = nx.adjacency_matrix(G).astype(float)
    A = A.tocsr()
    densest_onehop_indicator, densest_onehop_subgraph = utils.densest_onehop_neighborhood(G) 
    
    GreedyOQC_R = peeling.GreedyOQC(G, 5, alpha)
    GreedyOQC_indicator = utils.indicator_from_subgraph(G, GreedyOQC_R[1])
    GreedyOQC_density = LL_optimization.f_a(GreedyOQC_indicator,A,alpha)

    # Initialization
    if initialization=='S_oh':
        x0 = densest_onehop_indicator
    elif initialization=='S_greedy':
        x0 = GreedyOQC_indicator

    # Bounds: x in [0, 1]
    bounds = [(0, 1)] * n

    result, fun_values_, is_binary_ = LL_optimization.homotopy_LL(A, alpha, lam0=lam, mu0=mu, x0=x0, 
                                                              method=method, bounds=bounds, tolerance=tolerance, 
                                                              max_iters=max_iters, dec_rate=dec_rate)

    # Print result
    # print("Optimized parameters:", result.x)
    print("Function value:", result.fun)
    print("Success:", result.success)
    print("Message:", result.message)

    nodes = list(G.nodes)
    selected_nodes = [node for i, node in enumerate(nodes) if result.x[i] >= 0.5]
    G_sub = G.subgraph(selected_nodes).copy()

    print('Optimization method')
    print("OQC-Density: ", peeling.compute_degree_function(G_sub.number_of_nodes(), G_sub.number_of_edges(), alpha=alpha))
    print("Avg-Degree Density: ", G_sub.number_of_edges()/G_sub.number_of_nodes())
    print("Number of Nodes: ", G_sub.number_of_nodes())

    OQC_density = peeling.compute_degree_function(G_sub.number_of_nodes(), G_sub.number_of_edges(), alpha=alpha)
    avg_deg_density = G_sub.number_of_edges()/G_sub.number_of_nodes()
    num_of_nodes = G_sub.number_of_nodes()

    variables_dict = {
        'result': result,
        'fun_values_': fun_values_,
        'is_binary_': is_binary_,
        'selected_nodes': selected_nodes,
        'OQC_density': OQC_density,
        'avg_deg_density': avg_deg_density,
        'num_of_nodes': num_of_nodes,
        'x0': x0,
        'GreedyOQC_indicator': GreedyOQC_indicator,
        'GreedyOQC_density': GreedyOQC_density
    }

    save_folder = logs_save_path+'/'+dataset_name
    save_path = save_folder+'/'+dataset_name+'_alpha'+'{:.2f}'.format(alpha)
    save_path = save_path+'_'+initialization
    save_path = save_path+'_iters'+str(max_iters)+'_log.npy'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)

if __name__ == '__main__':
    main()