"""
Unit tests for dataset_processor.py.

Tests cover image loading, validation, resizing, batch generation,
and graceful handling of corrupted or oversized inputs.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from part_a_indexer.dataset_processor import ImageProcessor
from part_a_indexer.utils.image_utils import (
    batch_generator,
    is_valid_image,
    path_to_id,
    pil_to_numpy,
    resize_image,
    validate_pil_image,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def valid_rgb_image() -> Image.Image:
    """Return a solid-colour 256×256 RGB PIL Image."""
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    return img


@pytest.fixture
def small_image() -> Image.Image:
    """Return a 10×10 image (below minimum acceptable size)."""
    return Image.new("RGB", (10, 10), color=(0, 0, 0))


@pytest.fixture
def temp_image_file(valid_rgb_image, tmp_path) -> Path:
    """Save a valid image to a temporary file and return its Path."""
    path = tmp_path / "test_image.jpg"
    valid_rgb_image.save(path, format="JPEG")
    return path


@pytest.fixture
def temp_corrupt_file(tmp_path) -> Path:
    """Create a file with garbage bytes (simulates corrupted image)."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"\xFF\xFE\x00\x01GARBAGE_DATA")
    return path


@pytest.fixture
def processor() -> ImageProcessor:
    """Return an ImageProcessor with default settings."""
    return ImageProcessor(target_size=224)


# ── ImageProcessor tests ──────────────────────────────────────────────────────


class TestImageProcessor:
    def test_init_default(self):
        """Processor initialises with correct default target size."""
        p = ImageProcessor()
        assert p.target_size == 224
        assert p.stats["processed"] == 0
        assert p.stats["skipped"] == 0

    def test_init_custom_size(self):
        """Processor respects custom target size."""
        p = ImageProcessor(target_size=336)
        assert p.target_size == 336

    def test_load_image_valid(self, processor, temp_image_file):
        """Valid image is loaded and returned as numpy array."""
        result = processor.load_image(temp_image_file)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_load_image_corrupted(self, processor, temp_corrupt_file):
        """Corrupted image returns None without raising."""
        result = processor.load_image(temp_corrupt_file)
        assert result is None
        assert processor.stats["skipped"] == 1

    def test_load_image_missing(self, processor, tmp_path):
        """Missing file returns None without raising."""
        result = processor.load_image(tmp_path / "nonexistent.jpg")
        assert result is None

    def test_load_pil_valid(self, processor, temp_image_file):
        """load_pil returns a PIL Image."""
        result = processor.load_pil(temp_image_file)
        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)

    def test_preprocess_batch_yields_batches(self, processor, tmp_path, valid_rgb_image):
        """Batch generator yields (images, ids) tuples."""
        # Create 5 valid images
        paths = []
        for i in range(5):
            p = tmp_path / f"img_{i:03d}.jpg"
            valid_rgb_image.save(p, format="JPEG")
            paths.append(p)

        batches = list(processor.preprocess_batch(paths, batch_size=3))
        # 5 images in batches of 3 → 2 batches
        assert len(batches) == 2
        imgs, ids = batches[0]
        assert len(imgs) == 3
        assert all(isinstance(img, Image.Image) for img in imgs)

    def test_preprocess_batch_skips_corrupt(self, processor, tmp_path, valid_rgb_image, temp_corrupt_file):
        """Corrupted files are silently skipped in batch."""
        good = tmp_path / "good.jpg"
        valid_rgb_image.save(good, format="JPEG")

        batches = list(processor.preprocess_batch([good, temp_corrupt_file], batch_size=32))
        assert len(batches) == 1
        imgs, ids = batches[0]
        assert len(imgs) == 1  # only the good image

    def test_stats_tracking(self, processor, temp_image_file, temp_corrupt_file):
        """Stats correctly track processed vs skipped counts."""
        processor.load_image(temp_image_file)   # processed
        processor.load_image(temp_corrupt_file)  # skipped

        assert processor.stats["processed"] == 1
        assert processor.stats["skipped"] == 1


# ── Utility function tests ─────────────────────────────────────────────────────


class TestImageUtils:
    def test_path_to_id(self):
        """path_to_id extracts the filename stem."""
        assert path_to_id("/data/fashion_001.jpg") == "fashion_001"
        assert path_to_id(Path("/tmp/img.png")) == "img"

    def test_validate_pil_image_valid(self, valid_rgb_image):
        """Valid 256×256 RGB image passes validation."""
        assert validate_pil_image(valid_rgb_image) is True

    def test_validate_pil_image_too_small(self, small_image):
        """Images smaller than MIN_IMAGE_DIM fail validation."""
        assert validate_pil_image(small_image) is False

    def test_resize_image(self, valid_rgb_image):
        """resize_image produces correct output dimensions."""
        resized = resize_image(valid_rgb_image, target_size=224)
        assert resized.size == (224, 224)
        assert resized.mode == "RGB"

    def test_pil_to_numpy_shape(self, valid_rgb_image):
        """pil_to_numpy returns correct shape and dtype."""
        arr = pil_to_numpy(valid_rgb_image)
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8

    def test_batch_generator_even(self):
        """batch_generator splits list evenly."""
        items = list(range(9))
        batches = list(batch_generator(items, batch_size=3))
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2]

    def test_batch_generator_remainder(self):
        """batch_generator handles remainder correctly."""
        items = list(range(10))
        batches = list(batch_generator(items, batch_size=3))
        assert len(batches) == 4
        assert batches[-1] == [9]

    def test_batch_generator_empty(self):
        """batch_generator yields nothing for empty input."""
        assert list(batch_generator([], batch_size=32)) == []
