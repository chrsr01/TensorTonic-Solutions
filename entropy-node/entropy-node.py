import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(-np.sum(probabilities * np.log2(probabilities)))


    
    