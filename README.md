# Graph Skeletonization

This project implements graph skeletonization using the Mapper algorithm [[1]](#1) from Topological Data Analysis for the purpose of circuit analysis. It applies a real-valued function on the nodes of the graph, which combines different topological lenses, each focusing on a specific feature to preserve in the output graph.

- $f_{depth}$: measures vertical position relative to the output node (e.g., distance from output), encouraging a treelike hierarchy;
- $f_{supernode}$: estimates how likely it is that v is a member of a candidate supernode (e.g., based on precomputed semantic clusters);
- $f_{importance}$: scores how influential v is on the output (e.g., based on a path from v to the output node; or in cross-layer transcoders, on the weight of the edge from v to the output node)

## Structure
- `data/`: Graph data loading utilities
- `lenses/`: Lens function implementations
- `mapper/`: Mapper algorithm implementation
- `training/`: Weight optimization modules
- `apply/`: Application to new graphs
- `experiments/`: Example usage scripts

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run experiment:
    - No training, use the current optimized weights: 
    ```
    python -m experiments.run_experiment --graph data/sample_graphs/[SAMPLE-FILE].json
    ```
    - Find optimal weights based on a set of graphs as train set:
    ```
    python -m experiments.run_experiment --train_data_dir data/training/ --do_train --graph data/sample_graphs/[SAMPLE-FILE].json
    ```

## References
<a id="1">[1]</a> 
Rosen, P., Hajij, M., & Wang, B. (2023, October). Homology-preserving multi-scale graph skeletonization using mapper on graphs. In 2023 Topological Data Analysis and Visualization (TopoInVis) (pp. 10-20). IEEE.
