from .base import LensFunction
from data.loaders import get_output_nodes
import networkx as nx
from collections import deque, defaultdict

class ImportanceLens(LensFunction):
    def __init__(self):
        self.cache = {}  # Cache for graph computations
    
    def __call__(self, G, node):
        # Check if we've computed scores for this graph
        if id(G) not in self.cache:
            self._compute_scores(G)
        
        # Return cached score for the node
        scores = self.cache[id(G)]
        return scores.get(node, 0.0)
    
    def _compute_scores(self, G):
        """Compute the score for all nodes based on paths to output nodes."""
        # Get output nodes
        output_nodes = get_output_nodes(G)
        
        # Initialize scores
        scores = defaultdict(float)
        
        # For each output node, compute shortest paths to it
        for output in output_nodes:
            scores[output] = 1000 # set the importance of output nodes to a big number

            # Use BFS to get shortest path lengths from all nodes to this output
            lengths, paths = nx.single_target_shortest_path_length(G, output), nx.single_target_shortest_path(G, output)
            
            for node, path_length in lengths.items():
                if path_length == 0:
                    # Output node itself contributes 0 influence
                    continue
                
                path_nodes = paths[node]  # nodes along the path from node -> output
                influence_sum = 0
                for n in path_nodes:
                    if G.nodes[n].get('influence', 0.0) is not None: # is null for output nodes
                        influence_sum += G.nodes[n].get('influence', 0.0)
                # influence_sum = sum(G.nodes[n].get('influence', 0.0) for n in path_nodes)

                normalized_influence = influence_sum / path_length  # normalize by path length
                
                # Sum across all output nodes
                scores[node] += normalized_influence
                # if str(node).startswith("E") or str(node).startswith("27"):
                #     print(node, scores[node], path_length)
        
        print(len(G.nodes), len(list(scores.keys())), len(output_nodes))

        # Cache results
        self.cache[id(G)] = scores
