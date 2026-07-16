"""
Pytest configuration and shared fixtures for the Fashion Retrieval test suite.

This conftest.py is placed at the project root so that both
part_a_indexer/ and part_b_retriever/ tests share the same fixtures
and sys.path configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path for all tests
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute project root directory."""
    return _ROOT


@pytest.fixture(scope="session")
def sample_config() -> dict:
    """Return a minimal config dict for tests that need one."""
    return {
        "models": {
            "clip": {"name": "ViT-B/32", "embedding_dim": 512, "device": "cpu"},
            "fashion_clip": {"name": "patrickjohncyh/fashion-clip", "embedding_dim": 512},
        },
        "search": {
            "weights": {"semantic": 0.5, "fashion": 0.3, "attribute": 0.2},
            "top_k_candidates": 100,
            "final_top_k": 10,
            "confidence_threshold": 0.3,
        },
        "vector_db": {
            "dimension": 512,
            "metric": "cosine",
            "batch_size": 100,
        },
        "logging": {
            "level": "WARNING",
            "format": "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        },
    }
