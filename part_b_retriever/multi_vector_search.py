"""
Multi-Vector Search — Part B Retriever.

Performs parallel searches using CLIP and FashionCLIP text embeddings,
then combines the resulting score lists with configurable weights.

Weights (default):
  - CLIP (semantic):    50%
  - FashionCLIP:        30%
  - Attribute scores:   20%  (applied later in ranker.py)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from part_a_indexer.embedding_extractor import EmbeddingExtractor
from part_a_indexer.vector_storage import VectorStore

logger = logging.getLogger(__name__)


class MultiVectorSearch:
    """Search using CLIP and FashionCLIP text embeddings in parallel.

    Candidates from both searches are merged by taking their union and
    computing a weighted composite score. The attribute contribution
    (20%) is added later by ``ResultRanker``.

    Args:
        vector_store: Connected ``VectorStore`` instance.
        embedder: ``EmbeddingExtractor`` with loaded models.
        semantic_weight: CLIP score weight (default: 0.5).
        fashion_weight: FashionCLIP score weight (default: 0.3).
        top_k_candidates: Candidates fetched per Pinecone query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingExtractor,
        semantic_weight: float = 0.5,
        fashion_weight: float = 0.3,
        top_k_candidates: int = 100,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.fashion_weight = fashion_weight
        self.top_k_candidates = top_k_candidates

        logger.info(
            "MultiVectorSearch initialised (semantic=%.0f%%, fashion=%.0f%%, candidates=%d)",
            semantic_weight * 100,
            fashion_weight * 100,
            top_k_candidates,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multi-vector search for a natural language query.

        Args:
            query: Natural language search string.
            top_k: Maximum results to return after merging.
            metadata_filter: Optional Pinecone metadata filter dict.

        Returns:
            List of result dicts, sorted by combined score descending.
            Each dict has: ``id``, ``combined_score``, ``semantic_score``,
            ``fashion_score``, ``metadata``.
        """
        logger.debug("MultiVectorSearch.search: %r", query)

        # 1. Encode query with both encoders
        clip_embedding = self.embedder.encode_text_clip([query])[0]
        fashion_embedding = self.embedder.encode_text_fashion_clip([query])[0]

        # 2. Parallel Pinecone queries
        results_semantic = self._query_pinecone(clip_embedding, metadata_filter, namespace="")
        results_fashion = self._query_pinecone(fashion_embedding, metadata_filter, namespace="fashion_clip")

        logger.debug(
            "Pinecone returned: %d semantic, %d fashion candidates.",
            len(results_semantic),
            len(results_fashion),
        )

        # 3. Merge results with weighted scoring
        combined = self._combine_results(results_semantic, results_fashion)

        # 4. Sort and trim
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )
        return sorted_results[:top_k]

    def get_query_embeddings(self, query: str) -> Dict[str, np.ndarray]:
        """Return raw query embeddings without running a Pinecone search.

        Useful for caching or debugging.

        Args:
            query: Natural language query string.

        Returns:
            Dict with ``clip`` and ``fashion_clip`` numpy arrays.
        """
        return {
            "clip": self.embedder.encode_text_clip([query])[0],
            "fashion_clip": self.embedder.encode_text_fashion_clip([query])[0],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _query_pinecone(
        self,
        embedding: np.ndarray,
        filters: Optional[Dict],
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """Run a single Pinecone vector query.

        Args:
            embedding: Query vector, shape ``(512,)``.
            filters: Optional Pinecone metadata filter.
            namespace: Pinecone namespace to query.

        Returns:
            List of Pinecone result dicts (id, score, metadata).
        """
        try:
            return self.vector_store.query_by_vector(
                vector=embedding,
                top_k=self.top_k_candidates,
                filters=filters,
                include_metadata=True,
                namespace=namespace,
            )
        except Exception as exc:
            logger.error("Pinecone query failed: %s", exc)
            return []

    def _combine_results(
        self,
        results_semantic: List[Dict[str, Any]],
        results_fashion: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge two result lists with weighted score combination.

        Images appearing in only one list receive 0.0 for the
        missing score (not penalised — they may still rank well).

        Args:
            results_semantic: Results from CLIP query.
            results_fashion: Results from FashionCLIP query.

        Returns:
            Dict mapping ``image_id`` to a merged result dict.
        """
        # Build id → score maps
        semantic_map: Dict[str, Tuple[float, Dict]] = {
            r["id"]: (r["score"], r.get("metadata", {}))
            for r in results_semantic
        }
        fashion_map: Dict[str, float] = {
            r["id"]: r["score"]
            for r in results_fashion
        }

        # Union of all IDs
        all_ids = set(semantic_map.keys()) | set(fashion_map.keys())

        merged: Dict[str, Dict[str, Any]] = {}
        for img_id in all_ids:
            s_score, metadata = semantic_map.get(img_id, (0.0, {}))
            f_score = fashion_map.get(img_id, 0.0)

            combined_score = (
                self.semantic_weight * s_score
                + self.fashion_weight * f_score
            )

            merged[img_id] = {
                "id": img_id,
                "combined_score": round(combined_score, 6),
                "semantic_score": round(s_score, 6),
                "fashion_score": round(f_score, 6),
                "metadata": metadata,
            }

        return merged
