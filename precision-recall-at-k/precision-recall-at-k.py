def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    relevant_items = set(relevant)
    top_k = recommended[:k]
    hits = sum(item in relevant_items for item in top_k)
    precision = hits / k
    recall = hits / len(relevant_items)
    return [precision, recall]


    