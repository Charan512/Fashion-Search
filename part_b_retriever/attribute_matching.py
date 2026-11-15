"""
Attribute Matcher — Part B Retriever.

Scores how well a search result's image metadata matches the
structured components extracted from a user query.

Attribute matching contributes 20% to the final weighted score:
  final = 0.5 * semantic + 0.3 * fashion + 0.2 * attribute_match
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

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

        # ── Color & Compositional matching ──────────────────────────
        color_score = 0.0
        query_color_objs = query_components.get("colors", [])
        query_colors = [c["color"] for c in query_color_objs if c.get("color")]
        img_colors = [
            c.lower() for c in (
                (image_metadata.get("primary_colors") or [])
                + (image_metadata.get("secondary_colors") or [])
            ) if c
        ]
        img_items = [
            i.lower() for i in (image_metadata.get("clothing_items") or []) if i
        ]

        if query_colors:
            color_score, color_match, color_miss = self._score_list_match(
                query_colors, img_colors
            )
            matching.extend([f"color:{c}" for c in color_match])
            non_matching.extend([f"color:{c}" for c in color_miss])

            # Check compositional color-to-item binding if specified
            bound_hits = 0
            total_bound = 0
            for c_obj in query_color_objs:
                c_color = c_obj.get("color")
                c_item = c_obj.get("item")
                if c_color and c_item:
                    total_bound += 1
                    if c_color.lower() in img_colors and c_item.lower() in img_items:
                        bound_hits += 1
                        matching.append(f"bound:{c_color}+{c_item}")
                    else:
                        non_matching.append(f"unbound:{c_color}+{c_item}")

            if total_bound > 0:
                # Modulate color score with compositional co-presence
                composition_factor = bound_hits / total_bound
                color_score = 0.5 * color_score + 0.5 * composition_factor

        weighted_score += self.color_weight * color_score

        # ── Clothing matching ───────────────────────────────────────
        clothing_score = 0.0
        query_items = [c["item"] for c in query_components.get("clothing", []) if c.get("item")]
        if query_items:
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
            img_setting = str(image_metadata.get("setting") or "").lower()
            q_setting_clean = str(query_setting).lower()
            if img_setting and (q_setting_clean in img_setting or img_setting in q_setting_clean):
                setting_score = 1.0
                matching.append(f"setting:{query_setting}")
            elif img_setting:
                non_matching.append(f"setting:{query_setting}")

        weighted_score += self.setting_weight * setting_score

        # ── Formality matching ──────────────────────────────────────
        formality_score = 0.0
        context = query_components.get("context", {})
        query_formality = context.get("formality", 0.5)
        formality_specified = context.get("formality_specified", False) or abs(query_formality - 0.5) > 0.05
        img_formality = float(image_metadata.get("formality_score") or 0.5)

        if formality_specified:
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
            + (self.formality_weight if formality_specified else 0)
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
