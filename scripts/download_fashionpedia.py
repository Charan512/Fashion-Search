#!/usr/bin/env python3
"""
Download Fashionpedia dataset from HuggingFace Hub.

Caches images locally for faster re-indexing without re-downloading.
Saves images to <output_dir>/<split>/<image_id>.jpg

Usage:
    python scripts/download_fashionpedia.py --subset 1000 --output ./data/fashionpedia
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def download_fashionpedia(
    output_dir: str = "./data/fashionpedia",
    subset_size: int = 1000,
    split: str = "train",
) -> None:
    """Download Fashionpedia images to a local directory.

    Args:
        output_dir: Directory to save images.
        subset_size: Maximum images to download.
        split: Dataset split (``"train"`` or ``"validation"``).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    out_path = Path(output_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Fashionpedia dataset (split=%s, subset=%d)…", split, subset_size)
    dataset = load_dataset(
        "detection-datasets/fashionpedia",
        split=split,
        trust_remote_code=True,
    )

    if subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))

    saved = 0
    skipped = 0

    for idx, item in enumerate(dataset):
        image_id = str(item.get("image_id", f"img_{idx:06d}"))
        out_file = out_path / f"{image_id}.jpg"

        if out_file.exists():
            skipped += 1
            continue

        pil_img = item.get("image")
        if pil_img is None:
            logger.warning("Item %d has no image — skipping.", idx)
            continue

        try:
            pil_img.convert("RGB").save(out_file, format="JPEG", quality=90)
            saved += 1
        except Exception as exc:
            logger.warning("Could not save image %s: %s", image_id, exc)

        if (idx + 1) % 100 == 0:
            logger.info("Progress: %d/%d (saved=%d, skipped=%d)", idx + 1, subset_size, saved, skipped)

    logger.info("Download complete: %d saved, %d already existed.", saved, skipped)
    logger.info("Images saved to: %s", out_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fashionpedia dataset")
    parser.add_argument("--output", default="./data/fashionpedia", help="Output directory")
    parser.add_argument(
        "--subset",
        type=int,
        default=int(os.environ.get("DATASET_SUBSET_SIZE", 1000)),
        help="Number of images to download",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()

    download_fashionpedia(
        output_dir=args.output,
        subset_size=args.subset,
        split=args.split,
    )


if __name__ == "__main__":
    main()
