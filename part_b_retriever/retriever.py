"""
Fashion Retriever — Part B Orchestrator.

Central entry point for running end-to-end fashion image search.
Integrates all Part B components:
  QueryDecomposer → MultiVectorSearch → ResultRanker → SearchResult[]

Features:
  - In-memory LRU cache for query embeddings (avoids recomputing)
  - Graceful degradation if Pinecone is unavailable
  - Configurable search weights and top-k
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from part_a_indexer.embedding_extractor import EmbeddingExtractor
from part_a_indexer.utils.config_utils import get_env_var, load_config
from part_a_indexer.vector_storage import VectorStore
from part_b_retriever.attribute_matching import AttributeMatcher
from part_b_retriever.multi_vector_search import MultiVectorSearch
from part_b_retriever.query_processor import QueryDecomposer
from part_b_retriever.ranker import ResultRanker
from part_b_retriever.utils.explainability import ExplainabilityEngine, SearchResult

logger = logging.getLogger(__name__)


class FashionRetriever:
    """End-to-end fashion image retrieval system.

    Orchestrates the full search pipeline:
      1. Decompose query into structured components
      2. Encode query with CLIP + FashionCLIP
      3. Search Pinecone with both vectors
      4. Score and rank candidates with attribute matching
      5. Return top-k SearchResult objects with explanations

    Args:
        config: Optional pre-loaded config dict. Loaded from
            ``config.yaml`` if not provided.
        pinecone_api_key: Pinecone API key. Defaults to
            ``PINECONE_API_KEY`` env var.
        pinecone_index_name: Index name. Defaults to
            ``PINECONE_INDEX_NAME`` env var.
        device: Compute device (``"cuda"`` or ``"cpu"``).
        top_k_candidates: Candidates fetched per Pinecone query.
        semantic_weight: CLIP embedding weight.
        fashion_weight: FashionCLIP embedding weight.
        attribute_weight: Attribute match weight.
        confidence_threshold: Minimum score to include a result.
        diversity_enabled: Whether to apply diversity re-ranking.
        hard_constraints: Whether to enforce hard formality filters.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        device: Optional[str] = None,
        top_k_candidates: int = 100,
        semantic_weight: float = 0.5,
        fashion_weight: float = 0.3,
        attribute_weight: float = 0.2,
        confidence_threshold: float = 0.15,
        diversity_enabled: bool = True,
        hard_constraints: bool = True,
    ) -> None:
        self._config = config or self._load_config_safe()
        self._device = device or os.environ.get("DEVICE", "cpu")

        # Resolve Pinecone credentials
        self._api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY", "")
        self._index_name = pinecone_index_name or os.environ.get(
            "PINECONE_INDEX_NAME", "fashion-retrieval"
        )

        self._top_k_candidates = top_k_candidates
        self._semantic_weight = semantic_weight
        self._fashion_weight = fashion_weight
        self._attribute_weight = attribute_weight
        self._confidence_threshold = confidence_threshold
        self._diversity_enabled = diversity_enabled
        self._hard_constraints = hard_constraints

        # Components (initialised lazily)
        self._embedder: Optional[EmbeddingExtractor] = None
        self._vector_store: Optional[VectorStore] = None
        self._query_decomposer: Optional[QueryDecomposer] = None
        self._multi_vector_search: Optional[MultiVectorSearch] = None
        self._attribute_matcher: Optional[AttributeMatcher] = None
        self._ranker: Optional[ResultRanker] = None

        logger.info("FashionRetriever created (device=%s).", self._device)

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search for fashion images matching a natural language query.

        Args:
            query: Natural language search query, e.g.
                   ``"A red tie and white shirt in a formal office"``.
            top_k: Maximum number of results to return.

        Returns:
            List of ``SearchResult`` objects sorted by relevance,
            each with score breakdowns and explanations.

        Raises:
            ValueError: If query is empty.
            RuntimeError: If Pinecone connection fails.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        query = query.strip()
        logger.info("FashionRetriever.search: %r (top_k=%d)", query, top_k)
        t_start = time.time()

        try:
            # 1. Decompose query
            components = self.query_decomposer.decompose_query(query)

            # 2. Multi-vector search (semantic + fashion, returns candidates)
            candidates = self.multi_vector_search.search(
                query=query,
                top_k=self._top_k_candidates,
            )

            # 3. Re-rank with attribute matching
            results = self.ranker.rerank(
                candidates=candidates,
                query_components=components,
                top_k=top_k,
            )

        except Exception as exc:
            logger.error("Search failed: %s", exc, exc_info=True)
            raise

        elapsed = (time.time() - t_start) * 1000
        logger.info(
            "Search complete: %d results in %.0fms.", len(results), elapsed
        )
        return results

    def get_query_components(self, query: str) -> Dict[str, Any]:
        """Return structured query decomposition without running a search.

        Useful for debugging what the system extracted from a query.

        Args:
            query: Natural language query string.

        Returns:
            Decomposed components dict.
        """
        return self.query_decomposer.decompose_query(query)

    # ── Lazy component initialisation ─────────────────────────────────────────

    @property
    def embedder(self) -> EmbeddingExtractor:
        """CLIP + FashionCLIP embedding extractor, loaded on first use."""
        if self._embedder is None:
            self._embedder = EmbeddingExtractor(device=self._device)
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        """Pinecone vector store, connected on first use."""
        if self._vector_store is None:
            if not self._api_key:
                raise RuntimeError(
                    "PINECONE_API_KEY is not set. Copy .env.example to .env and fill in your key."
                )
            self._vector_store = VectorStore(
                api_key=self._api_key,
                index_name=self._index_name,
            )
        return self._vector_store

    @property
    def query_decomposer(self) -> QueryDecomposer:
        """Query decomposition engine (dict-based, no ML)."""
        if self._query_decomposer is None:
            self._query_decomposer = QueryDecomposer()
        return self._query_decomposer

    @property
    def multi_vector_search(self) -> MultiVectorSearch:
        """Multi-vector Pinecone search engine."""
        if self._multi_vector_search is None:
            self._multi_vector_search = MultiVectorSearch(
                vector_store=self.vector_store,
                embedder=self.embedder,
                semantic_weight=self._semantic_weight,
                fashion_weight=self._fashion_weight,
                top_k_candidates=self._top_k_candidates,
            )
        return self._multi_vector_search

    @property
    def attribute_matcher(self) -> AttributeMatcher:
        """Attribute matcher for scoring image metadata against query."""
        if self._attribute_matcher is None:
            self._attribute_matcher = AttributeMatcher()
        return self._attribute_matcher

    @property
    def ranker(self) -> ResultRanker:
        """Result ranker with hard constraints and diversity."""
        if self._ranker is None:
            self._ranker = ResultRanker(
                attribute_matcher=self.attribute_matcher,
                explainer=ExplainabilityEngine(
                    semantic_weight=self._semantic_weight,
                    fashion_weight=self._fashion_weight,
                    attribute_weight=self._attribute_weight,
                ),
                attribute_weight=self._attribute_weight,
                confidence_threshold=self._confidence_threshold,
                diversity_enabled=self._diversity_enabled,
                hard_constraints=self._hard_constraints,
            )
        return self._ranker

    @staticmethod
    def _load_config_safe() -> Dict[str, Any]:
        """Load config, returning an empty dict on failure."""
        try:
            return load_config()
        except Exception:
            return {}
