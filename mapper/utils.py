import numpy as np

def create_cover(points, n_intervals=10, overlap=0.2):
    min_val, max_val = np.min(points), np.max(points)
    range_val = max_val - min_val
    interval_length = range_val / (n_intervals - overlap * (n_intervals - 1))
    step = interval_length * (1 - overlap)
    
    intervals = []
    start = min_val
    for _ in range(n_intervals):
        end = start + interval_length
        intervals.append((start, end))
        start = start + step
    
    return intervals