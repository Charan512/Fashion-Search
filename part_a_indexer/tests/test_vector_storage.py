"""
Unit tests for vector_storage.py.

Uses mocked Pinecone clients so no live Pinecone connection is needed.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from part_a_indexer.vector_storage import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pinecone_index() -> MagicMock:
    """Return a mock Pinecone Index with sensible defaults."""
    index = MagicMock()

    # Mock query response
    match1 = MagicMock()
    match1.id = "img_001"
    match1.score = 0.95
    match1.metadata = {"primary_colors": ["red"], "formality_score": 0.9}

    match2 = MagicMock()
    match2.id = "img_002"
    match2.score = 0.88
    match2.metadata = {"primary_colors": ["blue"], "formality_score": 0.5}

    response = MagicMock()
    response.matches = [match1, match2]
    index.query.return_value = response

    # Mock stats
    stats = MagicMock()
    stats.total_vector_count = 42
    index.describe_index_stats.return_value = stats

    return index


@pytest.fixture
def vector_store(mock_pinecone_index) -> VectorStore:
    """Return a VectorStore with mocked Pinecone index."""
    store = VectorStore(
        api_key="test_key",
        index_name="test-index",
        dimension=512,
        metric="cosine",
    )
    store._index = mock_pinecone_index
    store._pc = MagicMock()
    return store


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Return a list of 5 sample vector records."""
    return [
        {
            "id": f"img_{i:03d}",
            "values": np.random.randn(512).astype(np.float32),
            "metadata": {
                "image_id": f"img_{i:03d}",
                "primary_colors": ["red"],
                "formality_score": 0.8,
                "setting": "indoor_office",
            },
        }
        for i in range(5)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestVectorStore:
    def test_query_returns_list(self, vector_store):
        """query_by_vector returns a list."""
        query = np.random.randn(512).astype(np.float32)
        results = vector_store.query_by_vector(query, top_k=10)
        assert isinstance(results, list)

    def test_query_result_structure(self, vector_store):
        """Each result has id and score keys."""
        query = np.random.randn(512).astype(np.float32)
        results = vector_store.query_by_vector(query, top_k=10)

        for r in results:
            assert "id" in r
            assert "score" in r

    def test_query_result_scores_range(self, vector_store):
        """Scores are in [0, 1] for cosine similarity."""
        query = np.random.randn(512).astype(np.float32)
        results = vector_store.query_by_vector(query, top_k=10)

        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_upsert_batch_calls_index(self, vector_store, sample_records):
        """upsert_batch calls index.upsert at least once."""
        vector_store.upsert_batch(sample_records, batch_size=10)
        assert vector_store._index.upsert.called

    def test_upsert_batch_chunks_correctly(self, vector_store, sample_records):
        """upsert_batch respects batch_size chunking (5 records, size=2 → 3 calls)."""
        vector_store.upsert_batch(sample_records, batch_size=2)
        # 5 records / 2 per batch = ceil(5/2) = 3 calls
        assert vector_store._index.upsert.call_count == 3

    def test_upsert_batch_handles_numpy_values(self, vector_store, sample_records):
        """upsert_batch converts numpy arrays to Python lists for serialisation."""
        # Should not raise a serialisation error
        vector_store.upsert_batch(sample_records, batch_size=100)

    def test_get_index_stats(self, vector_store):
        """get_index_stats returns expected keys."""
        stats = vector_store.get_index_stats()
        assert "total_vector_count" in stats
        assert "dimension" in stats
        assert "index_name" in stats
        assert stats["total_vector_count"] == 42

    def test_sanitise_record_numpy_conversion(self):
        """_sanitise_record converts numpy values to native Python types."""
        record = {
            "id": "test",
            "values": np.array([1.0, 2.0, 3.0]),
            "metadata": {
                "score": np.float32(0.95),
                "count": np.int64(5),
                "embedding": np.zeros(10),
                "label": "red",
            },
        }
        sanitised = VectorStore._sanitise_record(record)

        assert isinstance(sanitised["values"], list)
        assert isinstance(sanitised["metadata"]["score"], float)
        assert isinstance(sanitised["metadata"]["count"], int)
        assert isinstance(sanitised["metadata"]["embedding"], list)
        assert sanitised["metadata"]["label"] == "red"

    def test_delete_by_ids(self, vector_store):
        """delete_by_ids calls index.delete with the right IDs."""
        ids = ["img_001", "img_002"]
        vector_store.delete_by_ids(ids)
        vector_store._index.delete.assert_called_once_with(ids=ids)
