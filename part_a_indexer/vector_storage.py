"""
Vector Storage — Part A Indexer.

Provides a clean interface to the Pinecone vector database for:
  - Index creation (with idempotent check)
  - Batch upsert with automatic retry (tenacity)
  - Metadata-filtered vector queries
  - Index statistics
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Pinecone vector database interface.

    All vectors stored here use the CLIP global embedding (512-D)
    as the primary search vector. Other embeddings (FashionCLIP,
    scene) and all attributes are stored in metadata.

    Args:
        api_key: Pinecone API key.
        index_name: Name of the Pinecone index to create/connect to.
        dimension: Embedding dimensionality (default: 512 for CLIP).
        metric: Similarity metric — ``"cosine"`` recommended.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 512,
        metric: str = "cosine",
    ) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        self._index = None
        self._pc = None  # Pinecone client

        logger.info(
            "VectorStore configured (index=%s, dim=%d, metric=%s)",
            index_name,
            dimension,
            metric,
        )

    # ── Connection & setup ────────────────────────────────────────────────────

    @property
    def client(self):
        """Pinecone client, connected lazily."""
        if self._pc is None:
            self._connect()
        return self._pc

    @property
    def index(self):
        """Pinecone Index object, connected lazily."""
        if self._index is None:
            self._connect()
        return self._index

    def _connect(self) -> None:
        """Initialise Pinecone client and connect to the index."""
        try:
            from pinecone import Pinecone, ServerlessSpec  # noqa: F401

            pc = Pinecone(api_key=self.api_key)
            self._pc = pc
            self._index = pc.Index(self.index_name)
            logger.info("Connected to Pinecone index '%s'.", self.index_name)
        except Exception as exc:
            logger.error("Pinecone connection failed: %s", exc)
            raise

    def create_index_if_not_exists(self) -> None:
        """Create the Pinecone index if it does not already exist.

        Uses the Serverless free tier (``us-east-1`` / AWS).
        """
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=self.api_key)

            existing = [idx.name for idx in pc.list_indexes()]
            if self.index_name in existing:
                logger.info("Index '%s' already exists — skipping creation.", self.index_name)
            else:
                logger.info("Creating Pinecone index '%s'…", self.index_name)
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                # Wait until the index is ready
                for attempt in range(30):
                    status = pc.describe_index(self.index_name).status
                    if status.get("ready", False):
                        break
                    time.sleep(2)
                    logger.debug("Waiting for index to be ready (attempt %d)…", attempt + 1)

                logger.info("Index '%s' is ready.", self.index_name)

            self._pc = pc
            self._index = pc.Index(self.index_name)

        except Exception as exc:
            logger.error("Failed to create/connect Pinecone index: %s", exc)
            raise

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_batch(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 100,
        max_retries: int = 3,
        namespace: str = "",
    ) -> None:
        """Upsert a list of vector records to Pinecone in chunks.

        Each record must have: ``id`` (str), ``values`` (List[float]),
        ``metadata`` (Dict). Metadata values that are numpy arrays are
        automatically converted to Python lists.

        Args:
            records: List of vector record dicts.
            batch_size: Maximum records per API call.
            max_retries: Number of retry attempts per chunk on failure.

        Raises:
            RuntimeError: If a chunk fails after all retries.
        """
        total = len(records)
        uploaded = 0

        for start in range(0, total, batch_size):
            chunk = records[start : start + batch_size]
            # Sanitise metadata: convert numpy arrays to lists
            sanitised = [self._sanitise_record(r) for r in chunk]

            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    self.index.upsert(vectors=sanitised, namespace=namespace)
                    uploaded += len(chunk)
                    logger.debug("Upserted %d/%d vectors.", uploaded, total)
                    success = True
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "Upsert attempt %d/%d failed: %s — retrying in %ds.",
                        attempt,
                        max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)

            if not success:
                raise RuntimeError(
                    f"Upsert failed after {max_retries} attempts for chunk starting at {start}."
                )

        logger.info("Upsert complete: %d vectors stored.", uploaded)

    @staticmethod
    def _sanitise_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy values to Python native types for Pinecone serialisation."""
        sanitised = {
            "id": str(record["id"]),
            "values": (
                record["values"].tolist()
                if isinstance(record["values"], np.ndarray)
                else list(record["values"])
            ),
            "metadata": {},
        }
        for key, value in record.get("metadata", {}).items():
            if isinstance(value, np.ndarray):
                sanitised["metadata"][key] = value.tolist()
            elif isinstance(value, np.floating):
                sanitised["metadata"][key] = float(value)
            elif isinstance(value, np.integer):
                sanitised["metadata"][key] = int(value)
            else:
                sanitised["metadata"][key] = value
        return sanitised

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_by_vector(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """Find the *top_k* nearest vectors to *vector*.

        Args:
            vector: Query embedding of shape ``(D,)``.
            top_k: Number of results to return.
            filters: Pinecone metadata filter dict
                (e.g., ``{"setting": {"$eq": "indoor_office"}}``).
            include_metadata: Whether to include metadata in results.

        Returns:
            List of result dicts with keys ``id``, ``score``,
            and optionally ``metadata``.
        """
        query_list = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)

        response = self.index.query(
            vector=query_list,
            top_k=top_k,
            filter=filters,
            include_metadata=include_metadata,
            namespace=namespace,
        )

        results = []
        for match in response.matches:
            entry = {"id": match.id, "score": match.score}
            if include_metadata and match.metadata:
                entry["metadata"] = match.metadata
            results.append(entry)

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_index_stats(self) -> Dict[str, Any]:
        """Return statistics about the current index.

        Returns:
            Dict with ``total_vector_count``, ``dimension``, etc.
        """
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": self.dimension,
            "index_name": self.index_name,
        }

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to remove.
        """
        self.index.delete(ids=ids)
        logger.info("Deleted %d vectors from '%s'.", len(ids), self.index_name)
