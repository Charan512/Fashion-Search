"""
Unit tests for attribute_extractor.py.

All tests mock the CLIP model so they can run without GPU or
model downloads.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from part_a_indexer.attribute_extractor import (
    CLOTHING_ITEMS,
    COLOR_NAMES,
    SETTINGS,
    STYLE_CATEGORIES,
    AttributeExtractor,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image() -> Image.Image:
    """224×224 RGB PIL Image."""
    return Image.new("RGB", (224, 224), color=(180, 40, 40))  # reddish


def _make_extractor_with_mock_clip(top_score_index: int = 0) -> AttributeExtractor:
    """Return an AttributeExtractor whose CLIP is mocked.

    The mock always returns highest score for *top_score_index*.
    """
    extractor = AttributeExtractor(device="cpu")

    mock_model = MagicMock()
    mock_preprocess = MagicMock(return_value=MagicMock())

    def mock_encode_image(tensor):
        return torch.ones(1, 512)

    def mock_encode_text(tokens):
        n = tokens.shape[0]
        scores = torch.zeros(n)
        scores[top_score_index % n] = 10.0  # make one score dominant
        return scores.unsqueeze(0).expand(n, 512)

    mock_model.encode_image = mock_encode_image
    mock_model.encode_text = mock_encode_text

    extractor._clip_model = mock_model
    extractor._clip_preprocess = mock_preprocess
    return extractor


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestAttributeExtractor:
    def test_extract_colors_returns_list(self, dummy_image):
        """extract_colors returns a list of (str, float) tuples."""
        extractor = _make_extractor_with_mock_clip(top_score_index=0)

        # Patch _score_prompts to return deterministic probabilities
        probs = np.zeros(len(COLOR_NAMES), dtype=np.float32)
        probs[0] = 0.9  # first color = red
        probs[1] = 0.7  # second = blue

        with patch.object(extractor, "_score_prompts", return_value=probs):
            result = extractor.extract_colors(dummy_image)

        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_extract_colors_max_three(self, dummy_image):
        """extract_colors returns at most 3 colors."""
        extractor = _make_extractor_with_mock_clip()
        probs = np.ones(len(COLOR_NAMES), dtype=np.float32) * 0.9

        with patch.object(extractor, "_score_prompts", return_value=probs):
            result = extractor.extract_colors(dummy_image)

        assert len(result) <= 3

    def test_formality_score_range(self, dummy_image):
        """formality_score is always in [0, 1]."""
        extractor = _make_extractor_with_mock_clip()
        probs = np.random.uniform(0, 1, 10).astype(np.float32)

        with patch.object(extractor, "_score_prompts", return_value=probs):
            score = extractor.score_formality(dummy_image)

        assert 0.0 <= score <= 1.0

    def test_classify_setting_returns_string(self, dummy_image):
        """classify_setting returns a non-empty string."""
        extractor = _make_extractor_with_mock_clip()
        probs = np.random.uniform(0, 1, len(SETTINGS)).astype(np.float32)

        with patch.object(extractor, "_score_prompts", return_value=probs):
            result = extractor.classify_setting(dummy_image)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_classify_style_returns_valid_category(self, dummy_image):
        """classify_style returns a category from the known list."""
        extractor = _make_extractor_with_mock_clip()
        probs = np.zeros(len(STYLE_CATEGORIES), dtype=np.float32)
        probs[2] = 1.0  # index 2 = "casual"

        with patch.object(extractor, "_score_prompts", return_value=probs):
            result = extractor.classify_style(dummy_image)

        # Underscored version of a style category
        valid_styles = {s.replace(" ", "_") for s in STYLE_CATEGORIES}
        assert result in valid_styles

    def test_extract_all_attributes_keys(self, dummy_image):
        """extract_all_attributes returns dict with all required keys."""
        extractor = _make_extractor_with_mock_clip()

        def mock_score(img, prompts):
            n = len(prompts)
            probs = np.zeros(n, dtype=np.float32)
            probs[0] = 0.8
            return probs

        with patch.object(extractor, "_score_prompts", side_effect=mock_score):
            attrs = extractor.extract_all_attributes(dummy_image)

        required_keys = {
            "primary_colors",
            "secondary_colors",
            "clothing_items",
            "formality_score",
            "setting",
            "style_category",
        }
        assert required_keys.issubset(set(attrs.keys()))

    def test_formality_score_formal_dominant(self, dummy_image):
        """When formal prompts dominate, formality_score should be > 0.5."""
        extractor = _make_extractor_with_mock_clip()

        n_formal = 5
        n_casual = 5
        probs = np.array(
            [0.9] * n_formal + [0.1] * n_casual, dtype=np.float32
        )

        with patch.object(extractor, "_score_prompts", return_value=probs):
            score = extractor.score_formality(dummy_image)

        assert score > 0.5

    def test_confidence_threshold_filters_low_scores(self, dummy_image):
        """Colors/items below confidence_threshold are excluded."""
        extractor = AttributeExtractor(device="cpu", confidence_threshold=0.5)

        probs = np.array([0.3, 0.2, 0.1] + [0.0] * (len(COLOR_NAMES) - 3), dtype=np.float32)

        with patch.object(extractor, "_score_prompts", return_value=probs):
            result = extractor.extract_colors(dummy_image)

        # All scores are below threshold=0.5 → empty list
        assert result == []
