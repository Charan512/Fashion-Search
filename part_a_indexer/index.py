"""
Main Indexing Orchestrator — Part A.

Entry point for building the Fashionpedia vector index.
Coordinates ImageProcessor → EmbeddingExtractor → AttributeExtractor → VectorStore.

Usage:
    python -m part_a_indexer.index [--subset N] [--dry-run] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from part_a_indexer.attribute_extractor import AttributeExtractor
from part_a_indexer.dataset_processor import ImageProcessor, load_fashionpedia
from part_a_indexer.embedding_extractor import EmbeddingExtractor
from part_a_indexer.utils.config_utils import configure_logging, get_env_var, load_config
from part_a_indexer.utils.image_utils import path_to_id
from part_a_indexer.vector_storage import VectorStore

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("checkpoints/index_checkpoint.json")


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(batch_idx: int, processed: int) -> None:
    """Persist progress so indexing can be resumed after interruption.

    Args:
        batch_idx: Last completed batch index.
        processed: Total images successfully indexed so far.
    """
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "batch_idx": batch_idx,
        "processed": processed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_PATH, "w") as fh:
        json.dump(checkpoint, fh, indent=2)
    logger.debug("Checkpoint saved at batch %d (%d processed).", batch_idx, processed)


def load_checkpoint() -> Dict[str, Any]:
    """Load the latest checkpoint, or return defaults if none exists.

    Returns:
        Dict with ``batch_idx`` and ``processed`` keys.
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as fh:
            data = json.load(fh)
        logger.info(
            "Resuming from checkpoint: batch %d (%d images already indexed).",
            data["batch_idx"],
            data["processed"],
        )
        return data
    return {"batch_idx": 0, "processed": 0}


# ── Record builder ────────────────────────────────────────────────────────────


def build_records(
    image_ids: List[str],
    clip_embeddings,
    attributes: List[Dict],
    metadata_list: List[Dict],
) -> List[Dict[str, Any]]:
    """Assemble Pinecone-ready records from extracted data.

    Args:
        image_ids: List of string IDs.
        clip_embeddings: CLIP visual embeddings, shape ``(N, 512)``.
        attributes: List of attribute dicts per image.
        metadata_list: Source metadata dicts per image.

    Returns:
        List of Pinecone-ready record dicts.
    """
    records = []
    for i, image_id in enumerate(image_ids):
        src_meta = metadata_list[i] if i < len(metadata_list) else {}
        record = {
            "id": image_id,
            "values": clip_embeddings[i].tolist(),
            "metadata": {
                "image_id": image_id,
                # Attributes
                **attributes[i],
                # Source metadata
                "source": src_meta.get("source", "fashionpedia"),
                "split": src_meta.get("split", "train"),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        records.append(record)
    return records


# ── Main pipeline ─────────────────────────────────────────────────────────────


def build_index(
    subset_size: int = 1000,
    batch_size: int = 32,
    dry_run: bool = False,
    resume: bool = False,
) -> None:
    """Run the full image indexing pipeline.

    Steps:
      1. Load Fashionpedia images from HuggingFace Datasets
      2. For each batch: extract embeddings + attributes
      3. Assemble Pinecone records
      4. Upsert to Pinecone (unless ``dry_run=True``)
      5. Save a checkpoint every 10 batches

    Args:
        subset_size: Number of images to index.
        batch_size: Images processed per GPU batch.
        dry_run: If ``True``, skip Pinecone writes (for testing).
        resume: If ``True``, load and continue from the last checkpoint.
    """
    config = load_config()
    configure_logging(config)

    logger.info("=" * 60)
    logger.info("Fashion Retrieval — Indexer")
    logger.info("  Subset: %d images | Batch: %d | DryRun: %s", subset_size, batch_size, dry_run)
    logger.info("=" * 60)

    # Checkpoint state
    checkpoint = load_checkpoint() if resume else {"batch_idx": 0, "processed": 0}
    start_batch = checkpoint["batch_idx"]
    total_processed = checkpoint["processed"]

    # ── Load dataset ─────────────────────────────────────────
    images, metadata_list = load_fashionpedia(subset_size=subset_size)
    total_images = len(images)
    logger.info("Loaded %d images from Fashionpedia.", total_images)

    # ── Initialise components ─────────────────────────────────
    device = os.environ.get("DEVICE", config.get("models", {}).get("clip", {}).get("device", "cpu"))
    embedder = EmbeddingExtractor(device=device, batch_size=batch_size)
    attr_extractor = AttributeExtractor(device=device)

    # ── Vector store ──────────────────────────────────────────
    if not dry_run:
        api_key = get_env_var("PINECONE_API_KEY")
        index_name = get_env_var("PINECONE_INDEX_NAME", "fashion-retrieval")
        vector_store = VectorStore(api_key=api_key, index_name=index_name)
        vector_store.create_index_if_not_exists()
    else:
        vector_store = None
        logger.info("[DRY-RUN] Pinecone writes are disabled.")

    # ── Indexing loop ─────────────────────────────────────────
    processor = ImageProcessor(target_size=224)
    batch_count = 0
    t_start = time.time()

    for batch_idx_global, (batch_images, batch_ids) in enumerate(
        _generate_batches(images, metadata_list, batch_size)
    ):
        # Skip already-processed batches when resuming
        if batch_idx_global < start_batch:
            continue

        # Extract embeddings (all three views)
        clip_embs = embedder.extract_clip_embeddings(batch_images)
        fashion_embs = embedder.extract_fashion_clip_embeddings(batch_images)
        scene_embs = embedder.extract_scene_features(batch_images)

        # Extract attributes (per-image, sequential)
        attrs = []
        for img in batch_images:
            try:
                attrs.append(attr_extractor.extract_all_attributes(img))
            except Exception as exc:
                logger.warning("Attribute extraction failed for one image: %s", exc)
                attrs.append({
                    "primary_colors": [],
                    "secondary_colors": [],
                    "clothing_items": [],
                    "formality_score": 0.5,
                    "setting": "unknown",
                    "style_category": "unknown",
                })

        # Build and store records
        batch_meta = metadata_list[batch_idx_global * batch_size : (batch_idx_global + 1) * batch_size]
        records = build_records(batch_ids, clip_embs, attrs, batch_meta)

        if not dry_run and vector_store:
            vector_store.upsert_batch(records, namespace="")
            
            # Upsert FashionCLIP vectors into separate namespace
            fashion_records = []
            for i, image_id in enumerate(batch_ids):
                src_meta = metadata_list[batch_idx_global * batch_size + i] if (batch_idx_global * batch_size + i) < len(metadata_list) else {}
                record = {
                    "id": image_id,
                    "values": fashion_embs[i].tolist(),
                    "metadata": {
                        "image_id": image_id,
                        **attrs[i],
                        "source": src_meta.get("source", "fashionpedia"),
                    }
                }
                fashion_records.append(record)
            vector_store.upsert_batch(fashion_records, namespace="fashion_clip")

        total_processed += len(records)
        batch_count += 1
        percent = 100.0 * total_processed / total_images
        elapsed = time.time() - t_start
        rate = total_processed / elapsed if elapsed > 0 else 0

        logger.info(
            "Batch %d | %d/%d images (%.1f%%) | %.1f img/s",
            batch_count,
            total_processed,
            total_images,
            percent,
            rate,
        )

        # Save checkpoint every 10 batches
        if batch_count % 10 == 0:
            save_checkpoint(batch_idx_global + 1, total_processed)

    # Final checkpoint
    save_checkpoint(batch_count, total_processed)

    elapsed_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("✓ Indexing complete!")
    logger.info("  Total indexed: %d images in %.1fs", total_processed, elapsed_total)
    if not dry_run and vector_store:
        stats = vector_store.get_index_stats()
        logger.info("  Pinecone index now holds %d vectors.", stats.get("total_vector_count", "?"))
    logger.info("=" * 60)


def _generate_batches(images, metadata_list, batch_size):
    """Yield (image_batch, id_batch) pairs from the full image list."""
    from part_a_indexer.utils.image_utils import batch_generator

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start : batch_start + batch_size]
        batch_ids = [
            metadata_list[i].get("image_id", f"img_{i:06d}")
            for i in range(batch_start, min(batch_start + batch_size, len(images)))
        ]
        yield batch_images, batch_ids


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the indexer."""
    parser = argparse.ArgumentParser(
        description="Fashion Retrieval — Image Indexer (Part A)"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=int(os.environ.get("DATASET_SUBSET_SIZE", 1000)),
        help="Number of images to index (default: 1000; set DATASET_SUBSET_SIZE env var for default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Images processed per GPU batch (default: 32).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract embeddings but skip Pinecone writes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved checkpoint.",
    )

    args = parser.parse_args()
    build_index(
        subset_size=args.subset,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
