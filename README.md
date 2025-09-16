# Dense and Large Subgraph Detection via Lasry-Lions Double Envelopes - A Homotopy Approach

This is a code repository for our submission in "2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)".

## Abstract
l


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
