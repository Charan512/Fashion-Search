"""
Query Processor — Part B Retriever.

Parses natural language queries into structured components:
  - colors: list of {color, item, confidence} dicts
  - clothing: list of {item, confidence} dicts
  - context: {setting, formality, confidence}
  - style: style category string or None
  - descriptors: remaining adjective tokens

This module uses regex + dictionary lookups (no GPU required).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from part_b_retriever.utils.dictionaries import (
    CLOTHING_DICTIONARY,
    COLOR_DICTIONARY,
    SETTING_DICTIONARY,
    STYLE_DICTIONARY,
    estimate_formality_from_text,
    get_canonical_clothing,
    get_canonical_color,
    get_canonical_setting,
    get_canonical_style,
)

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Parse a natural language fashion query into structured components.

    No ML models are required — extraction is based on dictionary
    matching and lightweight regex patterns.

    Example:
        >>> qd = QueryDecomposer()
        >>> result = qd.decompose_query("A red tie and white shirt in a formal office")
        >>> result["colors"]
        [{'color': 'red', 'item': 'tie', 'confidence': 0.95},
         {'color': 'white', 'item': 'shirt', 'confidence': 0.95}]
    """

    def __init__(self) -> None:
        # Build flat synonym-to-canonical lookup tables for O(1) access
        self._color_synonyms: Dict[str, str] = {
            syn: canonical
            for canonical, synonyms in COLOR_DICTIONARY.items()
            for syn in synonyms
        }
        self._clothing_synonyms: Dict[str, str] = {
            syn: canonical
            for canonical, synonyms in CLOTHING_DICTIONARY.items()
            for syn in synonyms
        }
        logger.info("QueryDecomposer initialised.")

    # ── Public API ────────────────────────────────────────────────────────────

    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose a natural language query into structured components.

        Args:
            query: Natural language search query.

        Returns:
            Dict with keys: ``colors``, ``clothing``, ``context``,
            ``style``, ``descriptors``, ``raw_query``.
        """
        logger.debug("Decomposing query: %r", query)

        colors = self.extract_colors(query)
        clothing = self.extract_clothing(query)
        context = self.extract_context(query)
        style = self.extract_style(query)
        descriptors = self._extract_descriptors(query)

        result: Dict[str, Any] = {
            "raw_query": query,
            "colors": colors,
            "clothing": clothing,
            "context": context,
            "style": style,
            "descriptors": descriptors,
        }

        logger.debug("Decomposition result: %s", result)
        return result

    def extract_colors(self, query: str) -> List[Dict[str, Any]]:
        """Extract color mentions and their associated clothing items.

        Handles patterns like:
          - "red tie" → color=red, item=tie
          - "bright yellow raincoat" → color=yellow, item=raincoat
          - "red tie and white shirt" → two entries

        Args:
            query: Natural language query string.

        Returns:
            List of dicts with keys ``color``, ``item``, ``confidence``.
        """
        query_lower = query.lower()
        words = re.findall(r"[a-z-]+", query_lower)
        results: List[Dict[str, Any]] = []
        found_colors: set = set()

        for i, word in enumerate(words):
            canonical_color = self._color_synonyms.get(word)
            if canonical_color and canonical_color not in found_colors:
                found_colors.add(canonical_color)
                # Look for associated clothing item within ±3 words
                associated_item = self._find_nearby_clothing(words, i, window=3)
                results.append({
                    "color": canonical_color,
                    "item": associated_item,
                    "confidence": 0.95,
                })

        # Also check multi-word color phrases (e.g., "navy blue")
        for canonical_color, synonyms in COLOR_DICTIONARY.items():
            if canonical_color in found_colors:
                continue
            for syn in synonyms:
                if " " in syn and syn in query_lower:
                    # Multi-word match
                    results.append({
                        "color": canonical_color,
                        "item": None,
                        "confidence": 0.90,
                    })
                    found_colors.add(canonical_color)
                    break

        return results

    def extract_clothing(self, query: str) -> List[Dict[str, Any]]:
        """Extract clothing item mentions from the query.

        Args:
            query: Natural language query string.

        Returns:
            List of dicts with keys ``item`` and ``confidence``.
        """
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []
        found_items: set = set()

        # Check multi-word synonyms first (longer matches take priority)
        for canonical, synonyms in CLOTHING_DICTIONARY.items():
            synonyms_sorted = sorted(synonyms, key=len, reverse=True)
            for syn in synonyms_sorted:
                if syn in query_lower and canonical not in found_items:
                    found_items.add(canonical)
                    results.append({"item": canonical, "confidence": 0.95})
                    break

        return results

    def extract_context(self, query: str) -> Dict[str, Any]:
        """Extract location, setting, and formality from the query.

        Args:
            query: Natural language query string.

        Returns:
            Dict with keys ``setting`` (str or None),
            ``formality`` (float 0–1), ``confidence`` (float).
        """
        query_lower = query.lower()

        # Setting detection (longest keyword match wins)
        setting = None
        setting_confidence = 0.0
        best_match_len = 0
        
        for canonical, keywords in SETTING_DICTIONARY.items():
            for kw in keywords:
                if kw in query_lower and len(kw) > best_match_len:
                    setting = canonical
                    setting_confidence = 0.90
                    best_match_len = len(kw)

        # Formality estimation
        formality = estimate_formality_from_text(query)

        return {
            "setting": setting,
            "formality": formality,
            "confidence": setting_confidence,
        }

    def extract_style(self, query: str) -> Optional[str]:
        """Identify the overall style category mentioned in the query.

        Args:
            query: Natural language query string.

        Returns:
            Style category string (e.g., ``"business_formal"``)
            or ``None`` if no style is detected.
        """
        query_lower = query.lower()
        for canonical, keywords in STYLE_DICTIONARY.items():
            if any(kw in query_lower for kw in keywords):
                return canonical
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_nearby_clothing(
        self,
        words: List[str],
        color_idx: int,
        window: int = 3,
    ) -> Optional[str]:
        """Scan words near *color_idx* to find a clothing item.

        Args:
            words: Tokenised query words.
            color_idx: Index of the color word in *words*.
            window: How many words to check in each direction.

        Returns:
            Canonical clothing item name, or ``None``.
        """
        start = max(0, color_idx - window)
        end = min(len(words), color_idx + window + 1)

        for offset in range(1, window + 1):
            for direction in (1, -1):
                idx = color_idx + direction * offset
                if start <= idx < end:
                    candidate = words[idx]
                    canonical = self._clothing_synonyms.get(candidate)
                    if canonical:
                        return canonical
        return None

    def _extract_descriptors(self, query: str) -> List[str]:
        """Extract remaining descriptive adjectives not covered elsewhere.

        Args:
            query: Natural language query string.

        Returns:
            List of adjective tokens.
        """
        # Common fashion adjectives that aren't colors or clothing items
        descriptor_keywords = [
            "bright", "dark", "light", "vibrant", "muted", "bold",
            "pastel", "neon", "classic", "modern", "vintage", "retro",
            "slim", "fitted", "oversized", "loose", "tight",
            "long", "short", "mini", "maxi", "cropped",
        ]
        query_lower = query.lower()
        return [kw for kw in descriptor_keywords if kw in query_lower]
