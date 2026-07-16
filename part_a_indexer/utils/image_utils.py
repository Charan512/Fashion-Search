"""
Image utility helpers for the Fashion Retrieval Indexer.

Provides lightweight, reusable functions for image loading,
validation, path management, and batch generation — used by
``dataset_processor.py`` and ``embedding_extractor.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Minimum acceptable image dimension (pixels)
MIN_IMAGE_DIM = 32


def is_valid_image(path: str | Path) -> bool:
    """Return True if the file at *path* can be opened as an image.

    Args:
        path: File path to check.

    Returns:
        ``True`` if the file is a valid, readable image, ``False`` otherwise.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # catches truncated images
        return True
    except (UnidentifiedImageError, OSError, SyntaxError):
        return False


def load_pil_image(path: str | Path) -> Optional[Image.Image]:
    """Open an image file and convert it to RGB.

    Args:
        path: Path to the image file.

    Returns:
        PIL ``Image`` object in RGB mode, or ``None`` if loading fails.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, Exception) as exc:
        logger.warning("Skipped unreadable image %s: %s", path, exc)
        return None


def validate_pil_image(img: Image.Image) -> bool:
    """Check that a PIL Image meets minimum quality requirements.

    Args:
        img: PIL Image to validate.

    Returns:
        ``True`` if the image is usable, ``False`` otherwise.
    """
    w, h = img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        logger.debug("Image too small: %dx%d", w, h)
        return False
    if img.mode not in ("RGB", "RGBA", "L"):
        logger.debug("Unsupported image mode: %s", img.mode)
        return False
    return True


def resize_image(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Resize image to a square of *target_size* × *target_size* pixels.

    Uses high-quality Lanczos resampling.

    Args:
        img: Source PIL Image.
        target_size: Output pixel dimension (same for width and height).

    Returns:
        Resized PIL Image in RGB mode.
    """
    return img.convert("RGB").resize((target_size, target_size), Image.LANCZOS)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a uint8 numpy array (H, W, 3).

    Args:
        img: RGB PIL Image.

    Returns:
        numpy array of shape (H, W, 3), dtype uint8.
    """
    return np.array(img.convert("RGB"), dtype=np.uint8)


def path_to_id(path: str | Path) -> str:
    """Derive a stable image ID from its file path.

    Uses the stem (filename without extension) as the ID.

    Args:
        path: File path of the image.

    Returns:
        String ID, e.g. ``"fashion_001"`` from ``"/data/fashion_001.jpg"``.
    """
    return Path(path).stem


def get_image_paths(directory: str | Path, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")) -> List[Path]:
    """Recursively collect all image paths within *directory*.

    Args:
        directory: Root directory to search.
        extensions: Acceptable file extensions (lower-case).

    Returns:
        Sorted list of absolute ``Path`` objects.
    """
    directory = Path(directory)
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(directory.rglob(f"*{ext}"))
        paths.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def batch_generator(
    items: List,
    batch_size: int = 32,
) -> Generator[List, None, None]:
    """Yield successive fixed-size batches from *items*.

    Args:
        items: Iterable to split into batches.
        batch_size: Maximum number of items per batch.

    Yields:
        Lists of up to *batch_size* items.
    """
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
