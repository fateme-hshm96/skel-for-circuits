import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from .utils import create_cover
from .utils import prune_graph_by_lens_combination

class MapperPipeline:
    def __init__(self, lenses, weights, n_intervals=10, overlap=0.2):
        self.lenses = lenses
        self.weights = weights
        self.n_intervals = n_intervals
        self.overlap = overlap
    
    def _prune(self, G):
        G = prune_graph_by_lens_combination(G, self.lenses, self.weights)
        return G

    def _compute_lens(self, G):
        lens_values = {}
        print("Pipeline | pruned graph:", G)
        for node in G.nodes():
            combined = 0.0
            for lens, weight in zip(self.lenses, self.weights):
                combined += weight * lens(G, node)
            lens_values[node] = combined
        # print(lens_values.keys())
        return lens_values
    
    def __call__(self, G):
        G = self._prune(G)
        lens_values = self._compute_lens(G)
        points = np.array(list(lens_values.values())).reshape(-1, 1)
        intervals = create_cover(points, n_intervals=self.n_intervals, overlap=self.overlap)
        # print(intervals)

        clusters = []
        for interval in intervals:
            nodes_in_interval = [
                node for node, value in lens_values.items() 
                if interval[0] <= value < interval[1]
            ]
            if not nodes_in_interval:
                continue
                
            # Create subgraph and cluster
            subgraph = G.subgraph(nodes_in_interval)
            if subgraph.number_of_nodes() == 0:
                continue
                
            # Cluster connected components
            for comp in nx.connected_components(subgraph.to_undirected()):
                clusters.append(list(comp))
        
        
        # Create skeleton graph
        skeleton = nx.Graph()
        for i, cluster in enumerate(clusters):
            skeleton.add_node(i, nodes=cluster)

        # Add edges between clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                # if any(G.has_edge(u, v) for u in clusters[i] for v in clusters[j]):
                #     skeleton.add_edge(i, j)
                weight = sum(G[u][v].get('weight', 1.0) for u in clusters[i] for v in clusters[j] if G.has_edge(u, v))
                if weight > 0:
                    skeleton.add_edge(i, j, weight=weight)


        return skeleton