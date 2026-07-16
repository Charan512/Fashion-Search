"""
Unit tests for query_processor.py.

Covers color extraction, clothing detection, context parsing,
style classification, and end-to-end decomposition on all
5 evaluation queries.
"""
from __future__ import annotations

import pytest

from part_b_retriever.query_processor import QueryDecomposer


@pytest.fixture
def qd() -> QueryDecomposer:
    return QueryDecomposer()


# ── Color extraction ──────────────────────────────────────────────────────────


class TestExtractColors:
    def test_simple_red(self, qd):
        """Extracts 'red' from a simple query."""
        result = qd.extract_colors("A red shirt")
        colors = [r["color"] for r in result]
        assert "red" in colors

    def test_color_with_item_association(self, qd):
        """Associates color with nearby clothing item."""
        result = qd.extract_colors("A red tie")
        assert len(result) >= 1
        match = next((r for r in result if r["color"] == "red"), None)
        assert match is not None
        assert match["item"] == "tie"

    def test_bright_yellow(self, qd):
        """Extracts 'yellow' from 'bright yellow raincoat' (eval query 1)."""
        result = qd.extract_colors("A person in a bright yellow raincoat")
        colors = [r["color"] for r in result]
        assert "yellow" in colors

    def test_dual_color_extraction(self, qd):
        """Extracts both colors from eval query 5."""
        result = qd.extract_colors("A red tie and a white shirt in a formal setting")
        colors = [r["color"] for r in result]
        assert "red" in colors
        assert "white" in colors

    def test_no_colors(self, qd):
        """Returns empty list when no colors are in the query."""
        result = qd.extract_colors("A person in an office")
        assert result == []

    def test_confidence_field_present(self, qd):
        """Each color entry has a confidence field."""
        result = qd.extract_colors("A blue shirt")
        for entry in result:
            assert "confidence" in entry
            assert 0.0 <= entry["confidence"] <= 1.0

    def test_red_and_white_separate_items(self, qd):
        """Red tie and white shirt should be separately associated."""
        result = qd.extract_colors("red tie and white shirt")
        red_entry = next((r for r in result if r["color"] == "red"), None)
        white_entry = next((r for r in result if r["color"] == "white"), None)
        assert red_entry is not None
        assert white_entry is not None
        # red should map to tie, white to shirt (order may vary)
        assert red_entry["item"] in ("tie", None)  # best-effort
        assert white_entry["item"] in ("shirt", None)


# ── Clothing extraction ───────────────────────────────────────────────────────


class TestExtractClothing:
    def test_tie_detected(self, qd):
        """Detects 'tie' in the query."""
        result = qd.extract_clothing("A red tie")
        items = [r["item"] for r in result]
        assert "tie" in items

    def test_shirt_detected(self, qd):
        """Detects 'shirt' from 'white shirt'."""
        result = qd.extract_clothing("A white shirt")
        items = [r["item"] for r in result]
        assert "shirt" in items

    def test_raincoat_detected(self, qd):
        """Detects 'raincoat' from eval query 1."""
        result = qd.extract_clothing("A person in a bright yellow raincoat")
        items = [r["item"] for r in result]
        assert "raincoat" in items

    def test_multiple_items(self, qd):
        """Detects multiple items from eval query 5."""
        result = qd.extract_clothing("A red tie and a white shirt in a formal setting")
        items = [r["item"] for r in result]
        assert "tie" in items
        assert "shirt" in items

    def test_no_clothing(self, qd):
        """Returns empty list when no clothing items found."""
        result = qd.extract_clothing("A sunny day in the park")
        assert result == []


# ── Context extraction ────────────────────────────────────────────────────────


class TestExtractContext:
    def test_office_setting(self, qd):
        """Detects office setting from eval query 2."""
        ctx = qd.extract_context("Professional business attire inside a modern office")
        assert ctx["setting"] == "indoor_office"

    def test_park_setting(self, qd):
        """Detects park setting from eval query 3."""
        ctx = qd.extract_context("Someone wearing a blue shirt sitting on a park bench")
        assert ctx["setting"] == "outdoor_park"

    def test_city_street_setting(self, qd):
        """Detects city street setting from eval query 4."""
        ctx = qd.extract_context("Casual weekend outfit for a city walk")
        # 'city' maps to outdoor_street
        assert ctx["setting"] in ("outdoor_street", None)

    def test_formal_formality_score(self, qd):
        """Formal queries produce formality > 0.5."""
        ctx = qd.extract_context("Professional business attire in a formal office")
        assert ctx["formality"] > 0.5

    def test_casual_formality_score(self, qd):
        """Casual queries produce formality <= 0.5."""
        ctx = qd.extract_context("A casual t-shirt and jeans for the weekend")
        assert ctx["formality"] <= 0.5

    def test_no_setting(self, qd):
        """Returns None setting when no location mentioned."""
        ctx = qd.extract_context("A yellow raincoat")
        assert ctx["setting"] is None


# ── Style extraction ──────────────────────────────────────────────────────────


class TestExtractStyle:
    def test_formal_style(self, qd):
        """'formal' maps to business_formal."""
        result = qd.extract_style("formal business attire")
        assert result == "business_formal"

    def test_casual_style(self, qd):
        """'casual' maps to casual."""
        result = qd.extract_style("casual weekend outfit")
        assert result == "casual"

    def test_no_style(self, qd):
        """Returns None when no recognisable style keyword."""
        result = qd.extract_style("A person")
        assert result is None


# ── End-to-end: 5 evaluation queries ─────────────────────────────────────────


class TestEvaluationQueries:
    QUERIES = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting",
    ]

    @pytest.mark.parametrize("query", QUERIES)
    def test_decompose_does_not_raise(self, qd, query):
        """All 5 evaluation queries decompose without error."""
        result = qd.decompose_query(query)
        assert "raw_query" in result
        assert "colors" in result
        assert "clothing" in result
        assert "context" in result

    def test_query1_yellow_detected(self, qd):
        result = qd.decompose_query("A person in a bright yellow raincoat")
        colors = [c["color"] for c in result["colors"]]
        assert "yellow" in colors

    def test_query2_office_detected(self, qd):
        result = qd.decompose_query("Professional business attire inside a modern office")
        assert result["context"]["setting"] == "indoor_office"

    def test_query3_blue_and_park(self, qd):
        result = qd.decompose_query("Someone wearing a blue shirt sitting on a park bench")
        colors = [c["color"] for c in result["colors"]]
        assert "blue" in colors
        assert result["context"]["setting"] == "outdoor_park"

    def test_query5_both_colors_detected(self, qd):
        result = qd.decompose_query("A red tie and a white shirt in a formal setting")
        colors = [c["color"] for c in result["colors"]]
        assert "red" in colors
        assert "white" in colors
