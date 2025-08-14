from .base import LensFunction
from data.loaders import get_output_nodes
import networkx as nx
from collections import deque
import numpy as np

class ImportanceLens(LensFunction):
    def __init__(self):
        self.cache = {}  # Cache for graph computations
    
    def __call__(self, G, node):
        # Check if we've computed paths for this graph
        if id(G) not in self.cache:
            self._compute_shortest_paths(G)
        
        # Get cached results
        _, path_influence = self.cache[id(G)]
        
        # Return influence sum if path exists, else 0
        return path_influence.get(node, 0.0)
    
    def _compute_shortest_paths(self, G):
        """Precompute shortest paths to output nodes for all reachable nodes."""
        # Create reversed graph for path finding
        Gr = G.reverse()
        dist = {}
        path_influence = {}
        queue = deque()
        
        # Get output nodes
        output_nodes = get_output_nodes(G)
        # E.g. ['27_575_42', '27_3520_42', '27_7253_42', '27_948_42', '27_573_42', '27_752_42', '27_578_42', '27_2379_42']

        # Initialize BFS with output nodes
        for node in output_nodes:
            dist[node] = 0
            path_influence[node] = G.nodes[node]['influence']
            if not path_influence[node]: # output node
                path_influence[node] = 10000000
            queue.append(node)
        
        # BFS in reversed graph (which corresponds to paths to outputs)
        while queue:
            u = queue.popleft()
            current_dist = dist[u]
            current_influence = path_influence[u]
            
            # print(f"current_influence: {current_influence}")

            for v in Gr.neighbors(u):
                # Skip if we've already visited with a shorter path
                if v in dist and dist[v] <= current_dist + 1:
                    continue
                
                # Update distance and influence
                new_influence = G.nodes[v]['influence'] + current_influence
                dist[v] = current_dist + 1
                path_influence[v] = new_influence
                queue.append(v)
        
        # Cache results for this graph
        self.cache[id(G)] = (dist, path_influence)