# Graph Skeletonization

This project implements graph skeletonization using the Mapper algorithm from Topological Data Analysis. It is focused on Attribution Graphs and Circuit Analysis.

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
    - No training, use the current optimized weights: `python -m experiments.run_experiment --graph data/sample_graphs/[SAMPLE-FILE].json`
    - Find optimal weights based on a set of graphs as train set: `python -m experiments.run_experiment --train_data_dir data/training/ --do_train --graph data/sample_graphs/[SAMPLE-FILE].json `

