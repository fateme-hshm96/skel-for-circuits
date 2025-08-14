import os
import json
import numpy as np
import argparse
from data.loaders import load_graph, load_all_graphs, save_skeleton_json
from apply.apply_pipeline import apply_trained_weights
from lenses.depth import DepthLens
from lenses.supernode import SupernodeLens
from lenses.importance import ImportanceLens
from training.trainer import Trainer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():

    GRID_STEPS = 5

    parser = argparse.ArgumentParser(
        description="Apply trained mapper skeletonization to a graph"
    )
    parser.add_argument("--do_train", type=bool, default=False,
                        help="Train the weights for combining lenses")
    parser.add_argument("--graph", required=True,
                        help="Path to input graph JSON")
    parser.add_argument("--train_data_dir", type=str, 
                        help="Path to graph JSONs for training the weights")    
    parser.add_argument("--num_intervals", type=int, default=8,
                        help="Number of intervals in cover")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Overlap fraction in cover")
    parser.add_argument("--lambda_param", type=float, default=0.5,
                        help="Combining faithfulness and minimality")
    # parser.add_argument("--weights", required=True,
    #                     help="Path to JSON file with lens weights")
    # parser.add_argument("--supernode_threshold", type=float, default=0.7,
    #                     help="Threshold similarity of node labels to be grouped into supernodes.")
    # parser.add_argument("--cluster_method", choices=["l2","agglo"], default="l2",
    #                     help="Clustering method for mapper")
    # parser.add_argument("--out", required=True,
    #                     help="Path to save skeleton JSON")

    args = parser.parse_args()

    if args.do_train and not args.train_data_dir:
        parser.error("--train_data_dir is required when --do_train is specified.")
    
    # Initialize lenses
    lenses = [
        DepthLens(),
        SupernodeLens(),
        ImportanceLens()
    ]
    
    # ----------------------------
    # --- PHASE 1: TRAINING ---
    if args.do_train:
        print("---> Start Training...")
        # Load training graphs
        print("Loading training graphs...")
        train_graphs = load_all_graphs(args.train_data_dir)
        print(f"Loaded {len(train_graphs)} training graphs")


        # Train weights
        trainer = Trainer(train_graphs, lenses, args.num_intervals, args.overlap, args.lambda_param)

        # Generate weight grid (normalized to sum to 1)
        weights = np.linspace(0, 1, GRID_STEPS)
        weight_grid = []
        
        for w1 in weights:
            for w2 in weights:
                for w3 in weights:
                    total = w1 + w2 + w3
                    if total > 0:  # Avoid division by zero
                        normalized = [w1/total, w2/total, w3/total]
                        weight_grid.append(normalized)
        
        print(f"Generated {len(weight_grid)} weight combinations")
        
        # Perform grid search
        print("Starting grid search...")

        # best_weights, best_score = [0.2, 0.4, 0.4], 0.912

        best_weights, best_metrics, all_results = trainer.grid_search(weight_grid)

    # Save results
        results = {
            "best_weights": {
                "DepthLens": best_weights[0],
                "SupernodeLens": best_weights[1],
                "ImportanceLens": best_weights[2]
            },
            "best_metrics": best_metrics,
            "all_results": [
                {"weights": w, "metrics": m} 
                for w, m in all_results
            ],
            "config": {
                "lambda_param": args.lambda_param,
                "grid_steps": GRID_STEPS,
                "num_graphs": len(train_graphs)
            }
        }
        
        output_path = os.path.join("./training", "training_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved results to {output_path}")
        print(f"Best weights: {best_weights}")
        print(f"Faithfulness: {best_metrics['faithfulness']:.4f}")
        print(f"Minimality: {best_metrics['minimality']:.4f}")
        print(f"Composite Score: {best_metrics['score']:.4f}")


    # ----------------------------
    # --- PHASE 2: APPLICATION ---
    print("---> Skeletonizing...")
    
    # Load test graphs
    print("Loading test graphs...")
    test_graphs = [load_graph(args.graph)]
    print(f"Loaded {len(test_graphs)} test graphs")
    
    weights_path = "training/training_results.json"
    # Load best weights from training
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
        best_weights = weights_data["best_weights"]
    
    # Apply to each test graph
    print("Generating skeletons...")
    for i, graph in enumerate(test_graphs):
        print(f"Processing graph {i+1}/{len(test_graphs)}")
        
        # Apply mapper pipeline with trained weights
        skeleton = apply_trained_weights(
            graph=graph,
            lenses=lenses,
            trained_weights=best_weights,
            n_intervals=10,
            overlap=0.2
        )
        
        # Save skeleton
        output_path = os.path.join("data/outputs/", f"skeleton_{i}.json")
        save_skeleton_json(skeleton, output_path)
        print(f"Saved skeleton to {output_path}")
    
    print("\n=== PROCESS COMPLETE ===")
    print(f"Generated {len(test_graphs)} skeletons in data/outputs/")


if __name__ == "__main__":
    main()


# create lenses                                                     lenses/
# load graphs                                                       data/loaders.py
# create trainer                                                    training/trainer.py
#   trainer grid search
#       trainer objective
#           create MapperPipeline
#           call it
#               compute lenses
#               get intervals
#               create new skeleton
#           get faith an minimal score based on the skeleton
#           return combined score
#       decide on the current weights combination
#       return best weights combination
