import networkx as nx
from .base import LensFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SupernodeLens(LensFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2', k=5):
        self.model = SentenceTransformer(model_name)
        self.k = k
        
    def __call__(self, G, node):
        # Get all labels
        labels = [data['label'] for _, data in G.nodes(data=True)]
        embeddings = self.model.encode(labels)
        node_idx = list(G.nodes()).index(node)
        node_embedding = embeddings[node_idx].reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(node_embedding, embeddings)[0]
        top_k = np.partition(similarities, -self.k)[-self.k:]
        return np.mean(top_k)