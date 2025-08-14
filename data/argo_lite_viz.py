import json
import csv
import os
from typing import List, Dict


def load_json(path: str) -> Dict:
    """Load and return JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nodes_csv(nodes: List[Dict], path: str) -> None:
    """Save nodes to a CSV file for Argo Lite."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "feature", "layer", "activation", "influence"])
        for node in nodes:
            writer.writerow([
                node.get("node_id", ""),
                # node.get("feature", ""),  # feature, layer, and context id can be infereed from node id
                # node.get("layer", ""),
                # node.get("ctx_idx", ""),
                node.get("token_prob", ""),
                node.get("clerp", ""),
                node.get("activation", ""),
                node.get("influence", ""),
                node.get("feature_type", ""),
                node.get("is_target_logit", ""),
                
            ])

def save_edges_csv(edges: List[Dict], path: str) -> None:
    """Save edges to a CSV file for Argo Lite."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for edge in edges:
            writer.writerow([
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("weight", "")
            ])


def convert_json_to_csv(json_path: str) -> None:
    """Main function to load JSON and save nodes & edges CSVs with filenames based on input JSON."""
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    nodes_csv = f"{base_name}_nodes.csv"
    edges_csv = f"{base_name}_edges.csv"

    data = load_json(json_path)
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    save_nodes_csv(nodes, nodes_csv)
    save_edges_csv(links, edges_csv)

    print(f"✅ Saved {len(nodes)} nodes to {nodes_csv}")
    print(f"✅ Saved {len(links)} edges to {edges_csv}")


if __name__ == "__main__":
    # ---- CONFIG ----
    input_json_file = "data/sample_graphs/angelina.json"  # replace with your file
    convert_json_to_csv(input_json_file)
