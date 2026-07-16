"""
Unit tests for embedding_extractor.py.

All tests use CPU and mock CLIP model loading so they can run
without a GPU and without downloading actual model weights.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from part_a_indexer.utils.vector_utils import normalize_vector


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image() -> Image.Image:
    """224×224 RGB PIL Image filled with random pixels."""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_images(dummy_image) -> list:
    """List of 4 identical dummy images."""
    return [dummy_image] * 4


def _make_mock_clip(embed_dim: int = 512):
    """Return a mock CLIP model + preprocess pair."""
    model = MagicMock()
    preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))

    def mock_encode_image(inputs):
        batch = inputs.shape[0]
        return torch.randn(batch, embed_dim)

    def mock_encode_text(tokens):
        batch = tokens.shape[0]
        return torch.randn(batch, embed_dim)

    model.encode_image = mock_encode_image
    model.encode_text = mock_encode_text
    return model, preprocess


# ── EmbeddingExtractor tests ──────────────────────────────────────────────────


class TestEmbeddingExtractor:
    """Tests for EmbeddingExtractor using mocked models."""

    def _make_extractor(self):
        """Return an EmbeddingExtractor with mock CLIP loaded."""
        from part_a_indexer.embedding_extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(device="cpu", batch_size=2)

        mock_model, mock_preprocess = _make_mock_clip()
        extractor._clip_model = mock_model
        extractor._clip_preprocess = mock_preprocess

        # Scene projection
        import torch.nn as nn
        extractor._scene_projection = nn.Linear(512, 256, bias=False)
        extractor._scene_projection.eval()

        return extractor

    def test_extract_clip_embeddings_shape(self, dummy_images):
        """CLIP embeddings have correct shape (N, 512)."""
        extractor = self._make_extractor()
        result = extractor.extract_clip_embeddings(dummy_images)
        assert result.shape == (4, 512)

    def test_extract_clip_embeddings_dtype(self, dummy_images):
        """CLIP embeddings are float32."""
        extractor = self._make_extractor()
        result = extractor.extract_clip_embeddings(dummy_images)
        assert result.dtype == np.float32

    def test_clip_embeddings_normalised(self, dummy_images):
        """CLIP embeddings are L2-normalised (unit norm)."""
        extractor = self._make_extractor()
        result = extractor.extract_clip_embeddings(dummy_images)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_extract_scene_features_shape(self, dummy_images):
        """Scene features have correct shape (N, 256)."""
        extractor = self._make_extractor()
        result = extractor.extract_scene_features(dummy_images)
        assert result.shape == (4, 256)

    def test_fashion_clip_fallback_when_unavailable(self, dummy_images):
        """FashionCLIP returns zero array when model failed to load."""
        extractor = self._make_extractor()
        extractor._fashion_model = None
        extractor._fashion_processor = None

        result = extractor.extract_fashion_clip_embeddings(dummy_images)
        assert result.shape == (4, 512)
        np.testing.assert_array_equal(result, 0.0)

    def test_batch_extract_all_returns_dict(self, dummy_images):
        """batch_extract_all returns dict with all three keys."""
        extractor = self._make_extractor()
        result = extractor.batch_extract_all(dummy_images)

        assert "clip_global" in result
        assert "fashion_clip" in result
        assert "scene_embedding" in result

        assert result["clip_global"].shape == (4, 512)
        assert result["scene_embedding"].shape == (4, 256)

    def test_encode_text_clip_shape(self):
        """Text encoding returns shape (N, 512)."""
        extractor = self._make_extractor()

        with patch("clip.tokenize") as mock_tokenize:
            mock_tokenize.return_value = torch.zeros(2, 77, dtype=torch.long)
            result = extractor.encode_text_clip(["red shirt", "blue pants"])

        assert result.shape == (2, 512)

    def test_empty_images_list(self):
        """Extractor handles empty image list gracefully."""
        extractor = self._make_extractor()
        result = extractor.extract_clip_embeddings([])
        assert result.shape == (0, 512)

    def test_single_image(self, dummy_image):
        """Single image is processed correctly."""
        extractor = self._make_extractor()
        result = extractor.extract_clip_embeddings([dummy_image])
        assert result.shape == (1, 512)


# ── Vector utils tests ────────────────────────────────────────────────────────


class TestVectorUtils:
    def test_normalize_1d(self):
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_normalize_2d(self):
        v = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = normalize_vector(v)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_normalize_zero_vector(self):
        """Zero vector does not raise (avoids division by zero)."""
        v = np.zeros(10)
        result = normalize_vector(v)
        assert not np.any(np.isnan(result))
