from mapper.pipeline import MapperPipeline

def apply_trained_weights(graph, lenses, trained_weights, n_intervals=10, overlap=0.2):
    """
    Apply trained weights to a new graph using MapperPipeline.
    
    Args:
        graph: NetworkX graph to process
        lenses: List of lens functions
        trained_weights: Optimized weights from training
        n_intervals: Number of intervals for Mapper
        overlap: Overlap percentage for Mapper
    
    Returns:
        Skeleton graph (NetworkX Graph)
    """
    pipeline = MapperPipeline(
        lenses,
        trained_weights,
        n_intervals,
        overlap
    )
    return pipeline(graph)