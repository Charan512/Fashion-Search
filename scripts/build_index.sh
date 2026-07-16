#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# build_index.sh — Fashion Retrieval Indexer wrapper
# ─────────────────────────────────────────────────────────────
# Usage:
#   ./scripts/build_index.sh               # dev: 1000 images
#   ./scripts/build_index.sh --full        # production: 50000 images
#   ./scripts/build_index.sh --dry-run     # test without Pinecone writes
#   ./scripts/build_index.sh --resume      # resume from last checkpoint

set -euo pipefail

# Load .env if present
if [ -f ".env" ]; then
    set -o allexport
    source .env
    set +o allexport
    echo "✓ Loaded .env"
fi

SUBSET="${DATASET_SUBSET_SIZE:-1000}"
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --full) SUBSET=50000 ;;
        --dry-run) EXTRA_ARGS="$EXTRA_ARGS --dry-run" ;;
        --resume) EXTRA_ARGS="$EXTRA_ARGS --resume" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════"
echo "  Fashion Retrieval — Building Index"
echo "  Subset: $SUBSET images"
echo "  Args: $EXTRA_ARGS"
echo "═══════════════════════════════════════════════"

python3 -m part_a_indexer.index \
    --subset "$SUBSET" \
    $EXTRA_ARGS

echo ""
echo "✓ Indexing complete!"
