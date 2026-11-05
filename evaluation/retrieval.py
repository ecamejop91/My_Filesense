"""Pure binary document metrics. None means mathematically undefined."""


def validate_k(k):
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")


def unique_ranking(ranking):
    """Retain the first occurrence of each hashable document ID."""
    return list(dict.fromkeys(ranking))


def precision(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    return len(retrieved & relevant) / len(retrieved) if retrieved else 0.0


def recall(retrieved, relevant):
    relevant = set(relevant)
    return len(set(retrieved) & relevant) / len(relevant) if relevant else None


def precision_at_k(ranking, relevant, k):
    """Always divide by k, even when fewer than k documents are returned."""
    validate_k(k)
    return len(set(unique_ranking(ranking)[:k]) & set(relevant)) / k


def recall_at_k(ranking, relevant, k):
    validate_k(k)
    return recall(unique_ranking(ranking)[:k], relevant)
