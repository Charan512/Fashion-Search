"""
Dataset Processor — Part A Indexer.

Responsible for loading, validating, and preprocessing images
from the Fashionpedia dataset (or a local directory) into a form
suitable for downstream embedding extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from PIL import Image

from part_a_indexer.utils.image_utils import (
    batch_generator,
    get_image_paths,
    load_pil_image,
    path_to_id,
    pil_to_numpy,
    resize_image,
    validate_pil_image,
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Validate, load, and preprocess images for ML model input.

    Handles both a HuggingFace dataset and a local directory of images.
    Corrupted or undersized images are silently skipped with a warning log.

    Args:
        target_size: Output pixel dimension (images resized to
            ``target_size × target_size``). Default: 224.
    """

    def __init__(self, target_size: int = 224) -> None:
        self.target_size = target_size
        self._skipped: int = 0
        self._processed: int = 0
        logger.info("ImageProcessor initialized (target_size=%d)", target_size)

    # ── Public API ────────────────────────────────────────────

    def load_image(self, path: str | Path) -> Optional[np.ndarray]:
        """Load a single image from disk and preprocess it.

        Args:
            path: Absolute path to the image file.

        Returns:
            Preprocessed uint8 numpy array of shape (H, W, 3),
            or ``None`` if the image cannot be loaded.
        """
        pil_img = load_pil_image(path)
        if pil_img is None:
            self._skipped += 1
            return None

        if not validate_pil_image(pil_img):
            logger.warning("Invalid image (size/mode) — skipping: %s", path)
            self._skipped += 1
            return None

        resized = resize_image(pil_img, self.target_size)
        self._processed += 1
        return pil_to_numpy(resized)

    def load_pil(self, path: str | Path) -> Optional[Image.Image]:
        """Load and resize a PIL Image (used directly by CLIP preprocessors).

        Args:
            path: Absolute path to the image file.

        Returns:
            Resized PIL Image in RGB mode, or ``None`` on failure.
        """
        pil_img = load_pil_image(path)
        if pil_img is None or not validate_pil_image(pil_img):
            self._skipped += 1
            return None
        return resize_image(pil_img, self.target_size)

    def preprocess_batch(
        self,
        image_paths: List[str | Path],
        batch_size: int = 32,
    ) -> Generator[Tuple[List[Image.Image], List[str]], None, None]:
        """Yield batches of preprocessed PIL Images with their IDs.

        Corrupted images are skipped; the returned batch may be smaller
        than ``batch_size`` if some images in the window are invalid.

        Args:
            image_paths: List of absolute image file paths.
            batch_size: Maximum images per batch.

        Yields:
            ``(images, ids)`` where ``images`` is a list of PIL Images
            and ``ids`` is the corresponding list of string IDs.
        """
        total = len(image_paths)
        logger.info("Starting batch preprocessing of %d images", total)

        for batch_paths in batch_generator(image_paths, batch_size):
            images: List[Image.Image] = []
            ids: List[str] = []

            for path in batch_paths:
                pil_img = self.load_pil(path)
                if pil_img is not None:
                    images.append(pil_img)
                    ids.append(path_to_id(path))

            if images:
                yield images, ids
            else:
                logger.warning("Entire batch was invalid — all images skipped.")

        logger.info(
            "Preprocessing complete: %d processed, %d skipped.",
            self._processed,
            self._skipped,
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Return cumulative processing statistics.

        Returns:
            Dict with ``processed`` and ``skipped`` counts.
        """
        return {"processed": self._processed, "skipped": self._skipped}


# ── HuggingFace Dataset Loader ────────────────────────────────────────────────


def load_fashionpedia(
    subset_size: int = 1000,
    split: str = "train",
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    """Load Fashionpedia images from HuggingFace Datasets.

    Downloads the dataset on first call and caches it locally.
    Only ``subset_size`` items are loaded to keep memory manageable.

    Args:
        subset_size: Maximum number of images to load.
        split: Dataset split to use (``"train"`` or ``"validation"``).

    Returns:
        ``(images, metadata)`` where ``images`` is a list of PIL Images
        and ``metadata`` is a list of per-image dicts containing
        ``image_id``, ``source``, ``split``, and available labels.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    logger.info("Loading Fashionpedia dataset (split=%s, subset=%d)…", split, subset_size)

    # Detection-datasets/fashionpedia is publicly available on HF Hub
    dataset = hf_load_dataset(
        "detection-datasets/fashionpedia",
        split=split,
        trust_remote_code=True,
    )

    # Take a subset for dev speed
    if subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))

    images: List[Image.Image] = []
    metadata: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset):
        pil_img = item.get("image")
        if pil_img is None:
            logger.warning("Item %d has no 'image' field — skipping.", idx)
            continue

        # Ensure RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        image_id = str(item.get("image_id", f"fashionpedia_{idx:06d}"))
        images.append(pil_img)
        metadata.append(
            {
                "image_id": image_id,
                "source": "fashionpedia",
                "split": split,
                "width": pil_img.width,
                "height": pil_img.height,
            }
        )

        if (idx + 1) % 100 == 0:
            logger.info("Loaded %d / %d images…", idx + 1, subset_size)

    logger.info("Fashionpedia load complete: %d images ready.", len(images))
    return images, metadata
