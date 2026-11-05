"""Pure ranking metrics: grades 0..3, gain 2**grade-1, log2(rank+1)."""
from math import log2

from .retrieval import unique_ranking, validate_k


def validate_judgments(judgments):
    if any(type(g) is not int or not 0 <= g <= 3 for g in judgments.values()):
        raise ValueError("Relevance grades must be integers in 0..3")


def reciprocal_rank(ranking, relevant):
    """None for no relevant documents; 0 for relevant documents never retrieved."""
    relevant = set(relevant)
    if not relevant:
        return None
    return next((1.0 / i for i, doc in enumerate(unique_ranking(ranking), 1)
                 if doc in relevant), 0.0)


def mean_reciprocal_rank(rankings, relevant_sets):
    """Macro mean over eligible queries; exclude no-relevant queries explicitly."""
    rankings, relevant_sets = list(rankings), list(relevant_sets)
    if len(rankings) != len(relevant_sets):
        raise ValueError("Rankings and relevance sets must have equal lengths")
    values = [reciprocal_rank(r, rel) for r, rel in zip(rankings, relevant_sets)]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def dcg_at_k(ranking, judgments, k):
    validate_k(k)
    validate_judgments(judgments)
    return sum((2 ** judgments.get(doc, 0) - 1) / log2(i + 1)
               for i, doc in enumerate(unique_ranking(ranking)[:k], 1))


def ndcg_at_k(ranking, judgments, k):
    validate_k(k)
    validate_judgments(judgments)
    ideal = sorted(judgments, key=judgments.get, reverse=True)
    idcg = dcg_at_k(ideal, judgments, k)
    return dcg_at_k(ranking, judgments, k) / idcg if idcg else None
