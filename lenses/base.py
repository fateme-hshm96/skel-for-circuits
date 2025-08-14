import networkx as nx
from abc import ABC, abstractmethod

class LensFunction(ABC):
    @abstractmethod
    def __call__(self, G, node):
        """Compute lens value for a node in graph G."""
        pass