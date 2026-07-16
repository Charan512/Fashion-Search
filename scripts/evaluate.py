#!/usr/bin/env python3
"""
Evaluation Script — Run all 5 test queries and report results.

Usage:
    python scripts/evaluate.py [--top-k N] [--output-json PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from part_a_indexer.utils.config_utils import configure_logging, load_config
from part_b_retriever.retriever import FashionRetriever

EVALUATION_QUERIES = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
]


def run_evaluation(top_k: int = 10, output_json: str | None = None) -> None:
    """Run all 5 evaluation queries and print a results table.

    Args:
        top_k: Number of results to retrieve per query.
        output_json: Path to save results as JSON (optional).
    """
    config = load_config()
    configure_logging(config)

    print("=" * 70)
    print("Fashion Retrieval System — Evaluation")
    print(f"Top-K: {top_k}")
    print("=" * 70)

    retriever = FashionRetriever(
        device=os.environ.get("DEVICE", "cpu"),
        top_k_candidates=100,
    )

    all_results: List[Dict[str, Any]] = []

    for i, query in enumerate(EVALUATION_QUERIES, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)

        t0 = time.time()
        try:
            results = retriever.search(query, top_k=top_k)
            elapsed_ms = (time.time() - t0) * 1000

            print(f"  Found {len(results)} results in {elapsed_ms:.0f}ms")
            print()

            for rank, result in enumerate(results, 1):
                print(
                    f"  #{rank:2d} | {result.image_id:<30} | "
                    f"Score: {result.overall_score:.3f} | "
                    f"Semantic: {result.semantic_score:.3f} | "
                    f"Fashion: {result.fashion_score:.3f} | "
                    f"Attr: {result.attribute_score:.3f}"
                )
                if result.matching_attributes:
                    print(f"       Matched: {', '.join(result.matching_attributes[:3])}")
                print(f"       {result.explanation}")

            query_data = {
                "query_index": i,
                "query": query,
                "elapsed_ms": elapsed_ms,
                "result_count": len(results),
                "results": [r.to_dict() for r in results],
            }
            all_results.append(query_data)

        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_results.append({"query_index": i, "query": query, "error": str(exc)})

    print("\n" + "=" * 70)
    print("Evaluation complete.")

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Results saved to: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Fashion Retrieval System")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(top_k=args.top_k, output_json=args.output_json)


if __name__ == "__main__":
    main()
