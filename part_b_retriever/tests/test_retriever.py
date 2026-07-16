"""
Integration tests for retriever.py.

Tests the FashionRetriever with mocked Pinecone and ML models.
No live connections are made.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from part_b_retriever.retriever import FashionRetriever
from part_b_retriever.utils.explainability import SearchResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_mock_search_results(n: int = 5) -> List[Dict[str, Any]]:
    """Generate n mock search candidate dicts."""
    return [
        {
            "id": f"img_{i:03d}",
            "combined_score": 0.9 - i * 0.1,
            "semantic_score": 0.9 - i * 0.1,
            "fashion_score": 0.85 - i * 0.1,
            "metadata": {
                "primary_colors": ["red" if i % 2 == 0 else "blue"],
                "secondary_colors": ["white"],
                "clothing_items": ["tie", "shirt"],
                "formality_score": 0.85,
                "setting": "indoor_office",
                "style_category": "business_formal",
                "image_url": f"https://example.com/img_{i:03d}.jpg",
            },
        }
        for i in range(n)
    ]


@pytest.fixture
def mock_retriever() -> FashionRetriever:
    """Return a FashionRetriever with all external calls mocked."""
    retriever = FashionRetriever(
        pinecone_api_key="mock_key",
        pinecone_index_name="mock_index",
        device="cpu",
        confidence_threshold=0.0,
        hard_constraints=False,
    )

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.encode_text_clip.return_value = np.random.randn(1, 512).astype(np.float32)
    mock_embedder.encode_text_fashion_clip.return_value = np.random.randn(1, 512).astype(np.float32)
    retriever._embedder = mock_embedder

    # Mock vector store
    mock_vs = MagicMock()
    results_mock = _make_mock_search_results(5)
    mock_vs.query_by_vector.return_value = [
        {"id": r["id"], "score": r["semantic_score"], "metadata": r["metadata"]}
        for r in results_mock
    ]
    retriever._vector_store = mock_vs

    return retriever


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestFashionRetriever:
    def test_search_returns_search_results(self, mock_retriever):
        """search() returns a list of SearchResult objects."""
        results = mock_retriever.search("A red tie and white shirt", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_respects_top_k(self, mock_retriever):
        """search() returns at most top_k results."""
        results = mock_retriever.search("casual weekend outfit", top_k=3)
        assert len(results) <= 3

    def test_search_empty_query_raises(self, mock_retriever):
        """Empty query raises ValueError."""
        with pytest.raises(ValueError):
            mock_retriever.search("")

    def test_search_whitespace_query_raises(self, mock_retriever):
        """Whitespace-only query raises ValueError."""
        with pytest.raises(ValueError):
            mock_retriever.search("   ")

    def test_results_have_score_fields(self, mock_retriever):
        """Each result has required score fields."""
        results = mock_retriever.search("bright yellow raincoat", top_k=5)
        for r in results:
            assert hasattr(r, "overall_score")
            assert hasattr(r, "semantic_score")
            assert hasattr(r, "fashion_score")
            assert hasattr(r, "attribute_score")

    def test_results_have_explanation(self, mock_retriever):
        """Each result has a non-empty explanation."""
        results = mock_retriever.search("formal office attire", top_k=5)
        for r in results:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 0

    def test_get_query_components_returns_dict(self, mock_retriever):
        """get_query_components() returns expected structure."""
        components = mock_retriever.get_query_components(
            "A red tie and white shirt in a formal office"
        )
        assert "colors" in components
        assert "clothing" in components
        assert "context" in components
        assert "style" in components


# ── Evaluation query tests ────────────────────────────────────────────────────


class TestEvaluationQueries:
    """Ensure all 5 evaluation queries run without errors."""

    QUERIES = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting",
    ]

    @pytest.mark.parametrize("query", QUERIES)
    def test_evaluation_query_runs(self, mock_retriever, query):
        """Each evaluation query completes without error."""
        results = mock_retriever.search(query, top_k=10)
        assert isinstance(results, list)

    @pytest.mark.parametrize("query", QUERIES)
    def test_evaluation_query_components(self, mock_retriever, query):
        """Each evaluation query decomposes without error."""
        components = mock_retriever.get_query_components(query)
        assert "raw_query" in components
        assert components["raw_query"] == query
