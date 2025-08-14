def faithfulness_score(full_graph, skeleton_graph):
    """Calculate faithfulness as node coverage percentage."""
    skeleton_nodes = set()
    for _, data in skeleton_graph.nodes(data=True):
        skeleton_nodes.update(data['nodes'])
    
    return len(skeleton_nodes) / full_graph.number_of_nodes()

def minimality_score(skeleton_graph):
    """Calculate minimality as inverse of skeleton size."""
    n_nodes = skeleton_graph.number_of_nodes()
    return 1.0 / n_nodes if n_nodes > 0 else 0.0