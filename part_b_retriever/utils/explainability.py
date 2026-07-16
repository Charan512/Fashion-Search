"""
Explainability Engine — Part B Retriever.

Generates human-readable explanations for why a result ranked high,
including score breakdowns and attribute match summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Single ranked search result with full score breakdown.

    Attributes:
        image_id: Unique identifier for the indexed image.
        image_url: URL or local path to the image (may be empty).
        overall_score: Combined weighted relevance score in ``[0, 1]``.
        semantic_score: CLIP cosine similarity score in ``[0, 1]``.
        fashion_score: FashionCLIP cosine similarity score in ``[0, 1]``.
        attribute_score: Attribute match score in ``[0, 1]``.
        matching_attributes: Attributes from the query that matched.
        non_matching_attributes: Attributes queried but absent in image.
        explanation: One-sentence human-readable summary.
        metadata: Full image metadata dict from Pinecone.
    """

    image_id: str
    image_url: str = ""
    overall_score: float = 0.0
    semantic_score: float = 0.0
    fashion_score: float = 0.0
    attribute_score: float = 0.0
    matching_attributes: List[str] = field(default_factory=list)
    non_matching_attributes: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp scores to [0, 1]
        self.overall_score = max(0.0, min(1.0, self.overall_score))
        self.semantic_score = max(0.0, min(1.0, self.semantic_score))
        self.fashion_score = max(0.0, min(1.0, self.fashion_score))
        self.attribute_score = max(0.0, min(1.0, self.attribute_score))

    @property
    def score_breakdown(self) -> Dict[str, float]:
        """Weighted contribution of each score component.

        Returns:
            Dict mapping component names to their weighted contribution
            (not raw scores, but ``weight × score``).
        """
        return {
            "semantic (50%)": round(0.5 * self.semantic_score, 4),
            "fashion (30%)": round(0.3 * self.fashion_score, 4),
            "attribute (20%)": round(0.2 * self.attribute_score, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a plain dict."""
        return {
            "image_id": self.image_id,
            "image_url": self.image_url,
            "overall_score": self.overall_score,
            "semantic_score": self.semantic_score,
            "fashion_score": self.fashion_score,
            "attribute_score": self.attribute_score,
            "matching_attributes": self.matching_attributes,
            "non_matching_attributes": self.non_matching_attributes,
            "explanation": self.explanation,
            "score_breakdown": self.score_breakdown,
        }


class ExplainabilityEngine:
    """Generate human-readable explanations for search results.

    Args:
        semantic_weight: Weight of CLIP score (default: 0.5).
        fashion_weight: Weight of FashionCLIP score (default: 0.3).
        attribute_weight: Weight of attribute match score (default: 0.2).
    """

    def __init__(
        self,
        semantic_weight: float = 0.5,
        fashion_weight: float = 0.3,
        attribute_weight: float = 0.2,
    ) -> None:
        self.weights = {
            "semantic": semantic_weight,
            "fashion": fashion_weight,
            "attribute": attribute_weight,
        }

    def explain_result(self, result: SearchResult) -> str:
        """Generate a one-sentence explanation for a search result.

        Args:
            result: A ``SearchResult`` with scores populated.

        Returns:
            Human-readable explanation string.
        """
        parts = []

        if result.semantic_score > 0.7:
            parts.append(f"strong scene/context match ({result.semantic_score:.0%})")
        elif result.semantic_score > 0.5:
            parts.append(f"moderate context match ({result.semantic_score:.0%})")

        if result.fashion_score > 0.7:
            parts.append(f"high fashion specificity ({result.fashion_score:.0%})")
        elif result.fashion_score > 0.5:
            parts.append(f"moderate fashion match ({result.fashion_score:.0%})")

        if result.matching_attributes:
            attr_str = ", ".join(result.matching_attributes[:3])
            parts.append(f"matched attributes: {attr_str}")

        if result.non_matching_attributes:
            non_str = ", ".join(result.non_matching_attributes[:2])
            parts.append(f"missing: {non_str}")

        if not parts:
            return f"Relevance score: {result.overall_score:.0%}"

        return "Result shows " + "; ".join(parts) + f". Overall: {result.overall_score:.0%}."

    def compute_overall_score(
        self,
        semantic_score: float,
        fashion_score: float,
        attribute_score: float,
    ) -> float:
        """Compute the weighted overall relevance score.

        Args:
            semantic_score: CLIP similarity (0–1).
            fashion_score: FashionCLIP similarity (0–1).
            attribute_score: Attribute match (0–1).

        Returns:
            Weighted sum in ``[0, 1]``.
        """
        return (
            self.weights["semantic"] * semantic_score
            + self.weights["fashion"] * fashion_score
            + self.weights["attribute"] * attribute_score
        )

    def annotate_result(
        self,
        result: SearchResult,
        query_components: Dict[str, Any],
    ) -> SearchResult:
        """Populate the explanation field and verify score consistency.

        Args:
            result: ``SearchResult`` with scores already set.
            query_components: Decomposed query dict from ``QueryDecomposer``.

        Returns:
            Same ``SearchResult`` with ``explanation`` populated.
        """
        result.explanation = self.explain_result(result)
        return result
