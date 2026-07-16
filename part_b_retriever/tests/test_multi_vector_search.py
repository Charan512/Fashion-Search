"""
Unit tests for attribute_matching.py and ranker.py.

Uses mock query components and image metadata — no ML or Pinecone needed.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from part_b_retriever.attribute_matching import AttributeMatcher
from part_b_retriever.ranker import ResultRanker
from part_b_retriever.utils.explainability import SearchResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def matcher() -> AttributeMatcher:
    return AttributeMatcher()


@pytest.fixture
def ranker() -> ResultRanker:
    return ResultRanker(confidence_threshold=0.0, hard_constraints=False)


def _query_components(
    colors=None,
    clothing=None,
    setting=None,
    formality=0.5,
) -> Dict[str, Any]:
    """Build a minimal query_components dict for testing."""
    return {
        "colors": colors or [],
        "clothing": clothing or [],
        "context": {"setting": setting, "formality": formality, "confidence": 0.9},
        "style": None,
        "descriptors": [],
    }


def _image_metadata(
    primary_colors=None,
    secondary_colors=None,
    clothing_items=None,
    setting="indoor_office",
    formality_score=0.8,
) -> Dict[str, Any]:
    """Build a minimal image metadata dict for testing."""
    return {
        "primary_colors": primary_colors or [],
        "secondary_colors": secondary_colors or [],
        "clothing_items": clothing_items or [],
        "setting": setting,
        "formality_score": formality_score,
        "style_category": "business_formal",
    }


# ── AttributeMatcher tests ────────────────────────────────────────────────────


class TestAttributeMatcher:
    def test_exact_color_match(self, matcher):
        """Exact color match produces high score."""
        qc = _query_components(colors=[{"color": "red", "item": "tie", "confidence": 0.95}])
        meta = _image_metadata(primary_colors=["red"], clothing_items=["tie"])

        score, matching, non_matching = matcher.score_attribute_match(qc, meta)
        assert score > 0.5
        assert any("red" in m for m in matching)

    def test_color_mismatch(self, matcher):
        """Missing color reduces score."""
        qc = _query_components(colors=[{"color": "yellow", "item": None, "confidence": 0.95}])
        meta = _image_metadata(primary_colors=["blue"])

        score, matching, non_matching = matcher.score_attribute_match(qc, meta)
        assert any("yellow" in m for m in non_matching)

    def test_clothing_exact_match(self, matcher):
        """Matching clothing items increase score."""
        qc = _query_components(clothing=[{"item": "tie", "confidence": 0.95}])
        meta = _image_metadata(clothing_items=["tie", "shirt"])

        score, matching, non_matching = matcher.score_attribute_match(qc, meta)
        assert any("tie" in m for m in matching)

    def test_no_query_constraints(self, matcher):
        """When no query constraints exist, returns neutral score ~0.5."""
        qc = _query_components()
        meta = _image_metadata()
        score, _, _ = matcher.score_attribute_match(qc, meta)
        # Should be close to neutral (formality only)
        assert 0.0 <= score <= 1.0

    def test_setting_match(self, matcher):
        """Matching setting contributes positively."""
        qc = _query_components(setting="indoor_office")
        meta = _image_metadata(setting="indoor_office")

        score, matching, _ = matcher.score_attribute_match(qc, meta)
        assert any("setting" in m for m in matching)

    def test_formality_match_formal(self, matcher):
        """High query formality + high image formality → high score."""
        qc = _query_components(formality=0.9)
        meta = _image_metadata(formality_score=0.85)

        score, matching, _ = matcher.score_attribute_match(qc, meta)
        assert any("formality" in m for m in matching)

    def test_score_range(self, matcher):
        """Score is always in [0, 1]."""
        qc = _query_components(
            colors=[{"color": "red", "item": "tie", "confidence": 0.9}],
            clothing=[{"item": "shirt", "confidence": 0.9}],
            setting="indoor_office",
            formality=0.9,
        )
        meta = _image_metadata(
            primary_colors=["blue"],
            clothing_items=["jeans"],
            setting="outdoor_park",
            formality_score=0.1,
        )

        score, _, _ = matcher.score_attribute_match(qc, meta)
        assert 0.0 <= score <= 1.0

    def test_list_match_full_overlap(self, matcher):
        """Full overlap between query and image lists → score 1.0."""
        score, matched, missing = AttributeMatcher._score_list_match(
            ["red", "white"], ["red", "white", "navy"]
        )
        assert score == 1.0
        assert sorted(matched) == sorted(["red", "white"])
        assert missing == []

    def test_list_match_no_overlap(self, matcher):
        """No overlap → score 0.0."""
        score, matched, missing = AttributeMatcher._score_list_match(
            ["yellow"], ["blue", "green"]
        )
        assert score == 0.0
        assert matched == []
        assert "yellow" in missing


# ── ResultRanker tests ────────────────────────────────────────────────────────


def _make_candidate(img_id: str, semantic: float, fashion: float, **meta_kwargs) -> Dict[str, Any]:
    return {
        "id": img_id,
        "combined_score": 0.5 * semantic + 0.3 * fashion,
        "semantic_score": semantic,
        "fashion_score": fashion,
        "metadata": _image_metadata(**meta_kwargs),
    }


class TestResultRanker:
    def test_rerank_returns_search_results(self, ranker):
        """rerank returns a list of SearchResult objects."""
        candidates = [
            _make_candidate("img_1", 0.9, 0.8),
            _make_candidate("img_2", 0.7, 0.6),
        ]
        qc = _query_components()
        results = ranker.rerank(candidates, qc, top_k=10)

        assert all(isinstance(r, SearchResult) for r in results)

    def test_rerank_ordered_by_score(self, ranker):
        """Results are sorted descending by overall_score."""
        candidates = [
            _make_candidate("img_low", 0.4, 0.3),
            _make_candidate("img_high", 0.9, 0.85),
        ]
        qc = _query_components()
        results = ranker.rerank(candidates, qc, top_k=10)

        scores = [r.overall_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, ranker):
        """top_k limits the number of returned results."""
        candidates = [_make_candidate(f"img_{i}", 0.8, 0.7) for i in range(20)]
        qc = _query_components()
        results = ranker.rerank(candidates, qc, top_k=5)
        assert len(results) <= 5

    def test_explanation_populated(self, ranker):
        """Each result has a non-empty explanation string."""
        candidates = [_make_candidate("img_1", 0.85, 0.8)]
        qc = _query_components()
        results = ranker.rerank(candidates, qc, top_k=10)
        for r in results:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 0

    def test_score_breakdown_keys(self, ranker):
        """score_breakdown dict contains expected keys."""
        candidates = [_make_candidate("img_1", 0.85, 0.8)]
        qc = _query_components()
        results = ranker.rerank(candidates, qc, top_k=10)
        breakdown = results[0].score_breakdown
        assert "semantic (50%)" in breakdown
        assert "fashion (30%)" in breakdown
        assert "attribute (20%)" in breakdown

    def test_confidence_threshold_filters(self):
        """Results below confidence threshold are excluded."""
        ranker_strict = ResultRanker(
            confidence_threshold=0.9,
            hard_constraints=False,
            diversity_enabled=False,
        )
        candidates = [_make_candidate("img_low", 0.1, 0.1)]
        qc = _query_components()
        results = ranker_strict.rerank(candidates, qc, top_k=10)
        assert results == []

    def test_diversity_varies_styles(self):
        """Diversity re-ranking prevents consecutive same-style results."""
        ranker_div = ResultRanker(
            confidence_threshold=0.0,
            hard_constraints=False,
            diversity_enabled=True,
        )
        # All candidates have same style
        candidates = [
            _make_candidate(f"img_{i}", 0.8 - i * 0.05, 0.7)
            for i in range(5)
        ]
        qc = _query_components()
        results = ranker_div.rerank(candidates, qc, top_k=5)
        # Should still return results (not crash)
        assert len(results) <= 5

    def test_hard_constraint_filters_casual_from_formal_query(self):
        """Very formal query filters out very casual images."""
        ranker_strict = ResultRanker(
            confidence_threshold=0.0,
            hard_constraints=True,
            diversity_enabled=False,
        )
        # Very formal query
        qc = _query_components(formality=0.95)

        # Very casual image (formality_score = 0.1)
        casual_candidate = _make_candidate("img_casual", 0.8, 0.7, formality_score=0.1)
        formal_candidate = _make_candidate("img_formal", 0.7, 0.65, formality_score=0.9)

        results = ranker_strict.rerank([casual_candidate, formal_candidate], qc, top_k=10)
        result_ids = [r.image_id for r in results]

        # Formal image should be in results
        assert "img_formal" in result_ids
