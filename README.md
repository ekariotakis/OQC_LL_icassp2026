# Dense and Large Subgraph Detection via Lasry-Lions Double Envelopes - A Homotopy Approach

This is a code repository for our submission in "2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)".

## Abstract
Extracting large and dense subgraphs is a key task in graph mining applications. Pre-existing formulations are either overly restrictive (the maximum clique) or yield sparsely connected subgraphs (the densest subgraph problem). The optimal quasi-clique (OQC) problem allows extraction of large cliques and near-cliques of different sizes via a tunable parameter, but is NP-hard, with no principled optimization algorithm known. We propose a novel framework for OQC based on the Lasry-Lions (LL) double envelope for approximating OQC via a sequence of smooth subproblems within a homotopy framework. Experiments on real-world graphs show that our method outperforms the state-of-the-art greedy baseline for OQC, achieving better approximations of the size–density frontier and discovering large (near)-cliques that the baseline fails to detect.


## File Arrangement

Here we summarize all files present in this repo and their purpose.
```
+-- datasets/: 
    all the datasets used
+-- logs/: 
    some precomputed logs

+-- LL_optimization.py: 
    implementation of our method
+-- peeling.py: 
    implementation of the GreedyOQC peeling method

+-- run_LL.py:    
    example code to run our method

+-- utils.py: 
    some general utils used
+-- init_graph.py: 
    initialize graph and create protected group

+-- exec_run_LL.sh: 
    bash script example to execute our method with different initializations

+-- plot_delta_iters.ipynb - plot_delta_S.ipynb: 
    jupyter notebooks to reproduce Figure 1 and 2 of the paper
```
