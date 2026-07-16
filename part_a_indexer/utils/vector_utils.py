"""
Vector utility functions for the Fashion Retrieval System.

Provides normalization, similarity metrics, and score-combination
helpers used by both the Indexer and Retriever.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D or 2-D numpy array.

    Args:
        vector: Array of shape ``(D,)`` or ``(N, D)``.

    Returns:
        Unit-norm array of the same shape.
    """
    if vector.ndim == 1:
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-10)
    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    return vector / (norms + 1e-10)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Args:
        a: First vector, shape ``(D,)``.
        b: Second vector, shape ``(D,)``.

    Returns:
        Scalar similarity in ``[-1, 1]``.
    """
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return float(np.dot(a_norm, b_norm))


def batch_cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and a corpus of vectors.

    Args:
        query: Shape ``(D,)``.
        corpus: Shape ``(N, D)``.

    Returns:
        Similarity scores of shape ``(N,)``.
    """
    query_norm = normalize_vector(query)
    corpus_norm = normalize_vector(corpus)
    return corpus_norm @ query_norm  # (N,)


def combine_scores(
    scores_semantic: Dict[str, float],
    scores_fashion: Dict[str, float],
    scores_attribute: Dict[str, float],
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> Dict[str, Dict[str, float]]:
    """Combine three score dictionaries using weighted averaging.

    Any image present in at least one score dict is included; missing
    contributions default to 0.0.

    Args:
        scores_semantic: ``{image_id: score}`` from CLIP search.
        scores_fashion: ``{image_id: score}`` from FashionCLIP search.
        scores_attribute: ``{image_id: score}`` from attribute matching.
        weights: ``(w_semantic, w_fashion, w_attribute)`` summing to 1.0.

    Returns:
        ``{image_id: {combined, semantic, fashion, attribute}}``
    """
    w_s, w_f, w_a = weights
    all_ids = set(scores_semantic) | set(scores_fashion) | set(scores_attribute)

    combined: Dict[str, Dict[str, float]] = {}
    for img_id in all_ids:
        s = scores_semantic.get(img_id, 0.0)
        f = scores_fashion.get(img_id, 0.0)
        a = scores_attribute.get(img_id, 0.0)
        combined[img_id] = {
            "combined": w_s * s + w_f * f + w_a * a,
            "semantic": s,
            "fashion": f,
            "attribute": a,
        }

    return combined


def top_k_from_dict(
    score_dict: Dict[str, Dict[str, float]],
    top_k: int = 10,
    score_key: str = "combined",
) -> List[Tuple[str, Dict[str, float]]]:
    """Return the top-k entries from a score dict, sorted descending.

    Args:
        score_dict: Mapping of ``image_id → score_breakdown``.
        top_k: Number of results to return.
        score_key: Key within the breakdown to sort on.

    Returns:
        List of ``(image_id, breakdown)`` tuples.
    """
    return sorted(score_dict.items(), key=lambda x: x[1].get(score_key, 0.0), reverse=True)[:top_k]
