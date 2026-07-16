"""
Attribute Matcher — Part B Retriever.

Scores how well a search result's image metadata matches the
structured components extracted from a user query.

Attribute matching contributes 20% to the final weighted score:
  final = 0.5 * semantic + 0.3 * fashion + 0.2 * attribute_match
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AttributeMatcher:
    """Score a result's attribute metadata against query components.

    For each attribute category (colors, clothing, setting, formality),
    the matcher computes an exact/partial/no-match score and returns
    a normalised aggregate in ``[0, 1]``.

    Args:
        color_weight: Weight for color matching (default: 0.4).
        clothing_weight: Weight for clothing item matching (default: 0.3).
        setting_weight: Weight for setting/location matching (default: 0.15).
        formality_weight: Weight for formality score matching (default: 0.15).
    """

    def __init__(
        self,
        color_weight: float = 0.40,
        clothing_weight: float = 0.30,
        setting_weight: float = 0.15,
        formality_weight: float = 0.15,
    ) -> None:
        self.color_weight = color_weight
        self.clothing_weight = clothing_weight
        self.setting_weight = setting_weight
        self.formality_weight = formality_weight

    # ── Public API ────────────────────────────────────────────────────────────

    def score_attribute_match(
        self,
        query_components: Dict[str, Any],
        image_metadata: Dict[str, Any],
    ) -> Tuple[float, List[str], List[str]]:
        """Compute an attribute match score for an image.

        Args:
            query_components: Output from ``QueryDecomposer.decompose_query``.
            image_metadata: Pinecone metadata dict for the candidate image.

        Returns:
            ``(overall_score, matching_attributes, non_matching_attributes)``

            - ``overall_score``: float in ``[0, 1]``
            - ``matching_attributes``: list of matched attribute strings
            - ``non_matching_attributes``: list of unmatched query attributes
        """
        matching: List[str] = []
        non_matching: List[str] = []
        weighted_score = 0.0

        # ── Color matching ──────────────────────────────────────────
        color_score = 0.0
        query_colors = [c["color"] for c in query_components.get("colors", [])]
        if query_colors:
            img_colors = (
                image_metadata.get("primary_colors", [])
                + image_metadata.get("secondary_colors", [])
            )
            color_score, color_match, color_miss = self._score_list_match(
                query_colors, img_colors
            )
            matching.extend([f"color:{c}" for c in color_match])
            non_matching.extend([f"color:{c}" for c in color_miss])

        weighted_score += self.color_weight * color_score

        # ── Clothing matching ───────────────────────────────────────
        clothing_score = 0.0
        query_items = [c["item"] for c in query_components.get("clothing", [])]
        if query_items:
            img_items = image_metadata.get("clothing_items", [])
            clothing_score, cloth_match, cloth_miss = self._score_list_match(
                query_items, img_items
            )
            matching.extend([f"clothing:{c}" for c in cloth_match])
            non_matching.extend([f"clothing:{c}" for c in cloth_miss])

        weighted_score += self.clothing_weight * clothing_score

        # ── Setting matching ────────────────────────────────────────
        setting_score = 0.0
        query_setting = query_components.get("context", {}).get("setting")
        if query_setting:
            img_setting = image_metadata.get("setting", "")
            if img_setting and query_setting in img_setting:
                setting_score = 1.0
                matching.append(f"setting:{query_setting}")
            elif img_setting:
                non_matching.append(f"setting:{query_setting}")

        weighted_score += self.setting_weight * setting_score

        # ── Formality matching ──────────────────────────────────────
        formality_score = 0.0
        query_formality = query_components.get("context", {}).get("formality", 0.5)
        img_formality = float(image_metadata.get("formality_score", 0.5))

        # Score based on how close the two formality levels are
        formality_diff = abs(query_formality - img_formality)
        formality_score = max(0.0, 1.0 - formality_diff * 2)  # diff=0 → 1.0, diff=0.5 → 0.0

        if formality_score > 0.6:
            matching.append("formality:match")
        elif query_formality > 0.0:
            non_matching.append("formality:mismatch")

        weighted_score += self.formality_weight * formality_score

        # ── Normalise ───────────────────────────────────────────────
        total_weight = (
            (self.color_weight if query_colors else 0)
            + (self.clothing_weight if query_items else 0)
            + (self.setting_weight if query_setting else 0)
            + self.formality_weight
        )

        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        else:
            weighted_score = 0.5  # neutral when no constraints

        return round(weighted_score, 4), matching, non_matching

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _score_list_match(
        query_items: List[str],
        image_items: List[str],
    ) -> Tuple[float, List[str], List[str]]:
        """Score list overlap between query and image attributes.

        Scoring rules:
          - Exact match (present in image): 1.0 per item
          - Not present: 0.0 per item

        Args:
            query_items: Items required by the query.
            image_items: Items detected in the image.

        Returns:
            ``(score, matched_items, missing_items)``
        """
        if not query_items:
            return 0.5, [], []

        image_set = {item.lower() for item in image_items}
        matched = []
        missing = []

        for item in query_items:
            if item.lower() in image_set:
                matched.append(item)
            else:
                missing.append(item)

        score = len(matched) / len(query_items) if query_items else 0.5
        return score, matched, missing
