"""
Result Ranker — Part B Retriever.

Applies the final scoring, filtering, and diversity ranking
on candidate search results before returning to the user.

Pipeline:
  1. Apply attribute matching scores (20% weight)
  2. Compute final weighted combined score
  3. Apply hard constraints (formality threshold, setting filter)
  4. Apply confidence threshold (drop score < threshold)
  5. Apply diversity re-ranking (suppress near-duplicates)
  6. Build and return SearchResult objects
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from part_b_retriever.attribute_matching import AttributeMatcher
from part_b_retriever.utils.explainability import ExplainabilityEngine, SearchResult

logger = logging.getLogger(__name__)

# Formality threshold to enforce as a hard constraint
FORMALITY_HARD_THRESHOLD = 0.65


class ResultRanker:
    """Rank, filter, and annotate search candidates.

    Args:
        attribute_matcher: Scores attribute overlap between query and image.
        explainer: Generates score breakdown and explanation strings.
        attribute_weight: Weight of attribute matching in final score.
        confidence_threshold: Minimum overall score to include a result.
        diversity_enabled: Whether to apply style-diversity re-ranking.
        hard_constraints: Whether to enforce hard filters (formality, setting).
    """

    def __init__(
        self,
        attribute_matcher: Optional[AttributeMatcher] = None,
        explainer: Optional[ExplainabilityEngine] = None,
        attribute_weight: float = 0.20,
        confidence_threshold: float = 0.30,
        diversity_enabled: bool = True,
        hard_constraints: bool = True,
    ) -> None:
        self.attribute_matcher = attribute_matcher or AttributeMatcher()
        self.explainer = explainer or ExplainabilityEngine()
        self.attribute_weight = attribute_weight
        self.confidence_threshold = confidence_threshold
        self.diversity_enabled = diversity_enabled
        self.hard_constraints = hard_constraints

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        query_components: Dict[str, Any],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Rerank candidates into final SearchResult list.

        Args:
            candidates: Raw candidates from ``MultiVectorSearch.search``.
            query_components: Decomposed query from ``QueryDecomposer``.
            top_k: Maximum results to return.

        Returns:
            Sorted list of ``SearchResult`` objects.
        """
        # Step 1: Score candidates with attribute matching
        scored = self._score_candidates(candidates, query_components)

        # Step 2: Apply hard constraints
        if self.hard_constraints:
            scored = self._apply_hard_constraints(scored, query_components)

        # Step 3: Apply confidence threshold
        scored = [r for r in scored if r.overall_score >= self.confidence_threshold]

        # Step 4: Diversity re-ranking
        if self.diversity_enabled:
            scored = self._diversity_rerank(scored)

        # Step 5: Annotate with explanations
        final = []
        for result in scored[:top_k]:
            result = self.explainer.annotate_result(result, query_components)
            final.append(result)

        logger.debug("Reranked %d candidates → %d final results.", len(candidates), len(final))
        return final

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query_components: Dict[str, Any],
    ) -> List[SearchResult]:
        """Add attribute match scores and compute final overall scores.

        Args:
            candidates: Raw search result dicts with semantic/fashion scores.
            query_components: Decomposed query components.

        Returns:
            List of ``SearchResult`` objects sorted by overall_score descending.
        """
        results: List[SearchResult] = []

        for cand in candidates:
            metadata = cand.get("metadata", {})
            semantic_score = cand.get("semantic_score", 0.0)
            fashion_score = cand.get("fashion_score", 0.0)

            # Attribute matching (20%)
            attr_score, matching_attrs, non_matching_attrs = (
                self.attribute_matcher.score_attribute_match(query_components, metadata)
            )

            # Combine: semantic (already 0.5-weighted) + fashion (0.3) + attr (0.2)
            # Note: candidates have combined_score = 0.5*S + 0.3*F
            pre_combined = cand.get("combined_score", 0.0)
            overall = pre_combined + (attr_score * self.attribute_weight)

            image_url = metadata.get("image_url", "")

            result = SearchResult(
                image_id=cand["id"],
                image_url=image_url,
                overall_score=overall,
                semantic_score=semantic_score,
                fashion_score=fashion_score,
                attribute_score=attr_score,
                matching_attributes=matching_attrs,
                non_matching_attributes=non_matching_attrs,
                metadata=metadata,
            )
            results.append(result)

        return sorted(results, key=lambda r: r.overall_score, reverse=True)

    # ── Hard constraints ──────────────────────────────────────────────────────

    def _apply_hard_constraints(
        self,
        results: List[SearchResult],
        query_components: Dict[str, Any],
    ) -> List[SearchResult]:
        """Filter out results that violate hard query constraints.

        Hard constraints applied:
          - If query formality > ``FORMALITY_HARD_THRESHOLD``, only
            keep images with ``formality_score > 0.6``.
          - If query setting is specified, prefer matching settings
            (but don't exclude — graceful degradation).

        Args:
            results: Scored results to filter.
            query_components: Decomposed query components.

        Returns:
            Filtered list (may be same length if no violations).
        """
        context = query_components.get("context", {})
        query_formality = context.get("formality", 0.5)
        query_setting = context.get("setting")

        filtered: List[SearchResult] = []

        for result in results:
            meta = result.metadata
            img_formality = float(meta.get("formality_score", 0.5))

            # Hard formality filter: very formal queries require formal images
            if query_formality >= FORMALITY_HARD_THRESHOLD and img_formality < 0.55:
                logger.debug(
                    "Filtered out %s: formality %.2f < required %.2f",
                    result.image_id,
                    img_formality,
                    0.55,
                )
                continue

            filtered.append(result)

        # If hard filtering removed everything, fall back to original list
        if not filtered:
            logger.warning("Hard constraints removed all results — reverting to unconstrained.")
            return results

        return filtered

    # ── Diversity re-ranking ──────────────────────────────────────────────────

    def _diversity_rerank(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Promote diversity by avoiding consecutive same-style images.

        Uses a greedy MMR-like approach:
          - Always take the top-scoring unselected result
          - Apply a small penalty to results with the same style as
            the last selected result

        Args:
            results: Sorted results to diversify.

        Returns:
            Reordered list with improved style diversity.
        """
        if len(results) <= 3:
            return results

        diversified: List[SearchResult] = []
        remaining = list(results)
        last_style: Optional[str] = None

        while remaining:
            # Pick best candidate (with diversity penalty applied in-place)
            best_idx = 0
            best_score = -1.0

            for i, result in enumerate(remaining):
                style = result.metadata.get("style_category", "")
                score = result.overall_score
                # Small penalty if same style as last selected
                if style and style == last_style:
                    score *= 0.85

                if score > best_score:
                    best_score = score
                    best_idx = i

            selected = remaining.pop(best_idx)
            last_style = selected.metadata.get("style_category", "")
            diversified.append(selected)

        return diversified
