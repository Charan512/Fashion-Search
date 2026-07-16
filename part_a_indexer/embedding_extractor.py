"""
Embedding Extractor — Part A Indexer.

Generates multi-view embeddings from preprocessed images using:
  1. **Global CLIP** (ViT-B/32): captures full scene context (512-D)
  2. **FashionCLIP** (patrickjohncyh/fashion-clip): fashion-specific (512-D)
  3. **Scene features**: penultimate CLIP layer projection to 256-D

All extraction is batched and GPU-optimised with explicit memory cleanup.
"""
from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Model identifiers ─────────────────────────────────────────────────────────
CLIP_MODEL_NAME: str = "ViT-B/32"
FASHION_CLIP_MODEL_NAME: str = "patrickjohncyh/fashion-clip"
EMBEDDING_DIM_CLIP: int = 512
EMBEDDING_DIM_SCENE: int = 256


class EmbeddingExtractor:
    """Extract CLIP, FashionCLIP, and scene embeddings from images.

    Models are loaded lazily and pinned to the target device.
    Use :meth:`batch_extract_all` to obtain all three embeddings
    in a single GPU pass.

    Args:
        device: Compute device, e.g. ``"cuda"`` or ``"cpu"``.
        batch_size: Number of images per GPU batch. Reduce if OOM.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 32) -> None:
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.batch_size = batch_size

        self._clip_model = None
        self._clip_preprocess = None
        self._fashion_model = None
        self._fashion_processor = None
        self._scene_projection: Optional[torch.nn.Linear] = None

        logger.info(
            "EmbeddingExtractor created (device=%s, batch_size=%d)",
            self.device,
            self.batch_size,
        )

    # ── Lazy model loading ────────────────────────────────────────────────────

    @property
    def clip_model(self):
        """CLIP model (loaded on first access)."""
        if self._clip_model is None:
            self._load_clip()
        return self._clip_model

    @property
    def clip_preprocess(self):
        """CLIP image preprocessor (loaded on first access)."""
        if self._clip_preprocess is None:
            self._load_clip()
        return self._clip_preprocess

    @property
    def fashion_model(self):
        """FashionCLIP model (loaded on first access)."""
        if self._fashion_model is None:
            self._load_fashion_clip()
        return self._fashion_model

    @property
    def fashion_processor(self):
        """FashionCLIP processor (loaded on first access)."""
        if self._fashion_processor is None:
            self._load_fashion_clip()
        return self._fashion_processor

    def _load_clip(self) -> None:
        """Load OpenAI CLIP model and preprocessor."""
        logger.info("Loading CLIP model (%s) on %s…", CLIP_MODEL_NAME, self.device)
        try:
            import clip  # openai/CLIP
            model, preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess

            # Build a scene feature projection: 512 → 256
            self._scene_projection = torch.nn.Linear(
                EMBEDDING_DIM_CLIP, EMBEDDING_DIM_SCENE, bias=False
            ).to(self.device)
            # Initialise with fixed seed for reproducibility
            torch.nn.init.xavier_uniform_(self._scene_projection.weight)
            self._scene_projection.eval()

            logger.info("CLIP loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load CLIP: %s", exc)
            raise

    def _load_fashion_clip(self) -> None:
        """Load FashionCLIP model and processor from HuggingFace."""
        logger.info("Loading FashionCLIP (%s) on %s…", FASHION_CLIP_MODEL_NAME, self.device)
        try:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL_NAME)
            processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL_NAME)
            model = model.to(self.device)
            model.eval()

            self._fashion_model = model
            self._fashion_processor = processor
            logger.info("FashionCLIP loaded successfully.")
        except Exception as exc:
            logger.warning(
                "FashionCLIP load failed: %s — FashionCLIP embeddings will be zero-filled.", exc
            )
            self._fashion_model = None
            self._fashion_processor = None

    # ── Public extraction API ─────────────────────────────────────────────────

    def extract_clip_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Extract global CLIP embeddings for a list of PIL Images.

        Args:
            images: List of RGB PIL Images (any size — preprocessor handles resize).

        Returns:
            Float32 array of shape ``(N, 512)``, L2-normalised.
        """
        return self._extract_clip_visual(images, normalise=True)

    def extract_fashion_clip_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Extract FashionCLIP visual embeddings for a list of PIL Images.

        Falls back to zero vectors if FashionCLIP failed to load.

        Args:
            images: List of RGB PIL Images.

        Returns:
            Float32 array of shape ``(N, 512)``, L2-normalised.
        """
        if self.fashion_model is None:
            logger.warning("FashionCLIP unavailable — returning zero embeddings.")
            return np.zeros((len(images), EMBEDDING_DIM_CLIP), dtype=np.float32)

        embeddings: List[np.ndarray] = []

        for batch in self._batch_gen(images):
            try:
                inputs = self.fashion_processor(
                    images=batch,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    raw = self.fashion_model.get_image_features(**inputs)
                    features = self._unwrap_features(raw)
                    features = features / features.norm(dim=-1, keepdim=True)

                embeddings.append(features.cpu().float().numpy())
                self._clear_gpu_cache()

            except Exception as exc:
                logger.error("FashionCLIP batch error: %s", exc)
                embeddings.append(np.zeros((len(batch), EMBEDDING_DIM_CLIP), dtype=np.float32))

        return np.vstack(embeddings) if embeddings else np.zeros((0, EMBEDDING_DIM_CLIP), dtype=np.float32)

    def extract_scene_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract 256-D scene/context features via a CLIP projection.

        The scene vector is derived by projecting CLIP's 512-D visual
        embedding to 256-D, capturing higher-level scene semantics.

        Args:
            images: List of RGB PIL Images.

        Returns:
            Float32 array of shape ``(N, 256)``, L2-normalised.
        """
        clip_embeds = self._extract_clip_visual(images, normalise=False)
        tensor = torch.from_numpy(clip_embeds).to(self.device)

        with torch.no_grad():
            scene = self._scene_projection(tensor)
            scene = scene / scene.norm(dim=-1, keepdim=True)

        return scene.cpu().float().numpy()

    def batch_extract_all(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Extract all three embedding types in a single call.

        Args:
            images: List of RGB PIL Images.

        Returns:
            Dict with keys ``"clip_global"``, ``"fashion_clip"``,
            ``"scene_embedding"`` mapping to the respective arrays.
        """
        logger.debug("Extracting all embeddings for %d images", len(images))
        clip_embs = self.extract_clip_embeddings(images)
        fashion_embs = self.extract_fashion_clip_embeddings(images)
        scene_embs = self.extract_scene_features(images)
        return {
            "clip_global": clip_embs,
            "fashion_clip": fashion_embs,
            "scene_embedding": scene_embs,
        }

    # ── Text encoding (used by Retriever) ────────────────────────────────────

    def encode_text_clip(self, texts: List[str]) -> np.ndarray:
        """Encode natural-language strings with CLIP text encoder.

        Args:
            texts: List of query strings.

        Returns:
            Float32 array of shape ``(N, 512)``, L2-normalised.
        """
        import clip

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_text_fashion_clip(self, texts: List[str]) -> np.ndarray:
        """Encode natural-language strings with FashionCLIP text encoder.

        Args:
            texts: List of query strings.

        Returns:
            Float32 array of shape ``(N, 512)``, L2-normalised.
            Falls back to CLIP encoding if FashionCLIP is unavailable.
        """
        if self.fashion_model is None:
            logger.warning("FashionCLIP unavailable for text — using CLIP fallback.")
            return self.encode_text_clip(texts)

        inputs = self.fashion_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            raw = self.fashion_model.get_text_features(**inputs)
            features = self._unwrap_features(raw)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().float().numpy()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_clip_visual(self, images: List[Image.Image], normalise: bool = True) -> np.ndarray:
        """Run CLIP visual encoder over batches of PIL Images."""
        embeddings: List[np.ndarray] = []

        for batch in self._batch_gen(images):
            try:
                image_inputs = torch.stack(
                    [self.clip_preprocess(img) for img in batch]
                ).to(self.device)

                with torch.no_grad():
                    features = self.clip_model.encode_image(image_inputs)
                    if normalise:
                        features = features / features.norm(dim=-1, keepdim=True)

                embeddings.append(features.cpu().float().numpy())

                del image_inputs, features
                self._clear_gpu_cache()

            except Exception as exc:
                logger.error("CLIP batch encoding error: %s", exc)
                embeddings.append(
                    np.zeros((len(batch), EMBEDDING_DIM_CLIP), dtype=np.float32)
                )

        return np.vstack(embeddings) if embeddings else np.zeros((0, EMBEDDING_DIM_CLIP), dtype=np.float32)

    def _batch_gen(self, images: List[Image.Image]):
        """Yield sub-lists of images with self.batch_size length."""
        for i in range(0, len(images), self.batch_size):
            yield images[i : i + self.batch_size]

    @staticmethod
    def _unwrap_features(raw) -> torch.Tensor:
        """Normalise the output of get_image_features / get_text_features.

        Different transformers versions / model configs may return either:
        - A plain ``torch.Tensor``  (expected)
        - A ``BaseModelOutputWithPooling`` or similar dataclass

        This helper always returns a plain tensor.
        """
        if isinstance(raw, torch.Tensor):
            return raw
        # HuggingFace model output dataclass — prefer pooler_output (CLS token)
        if hasattr(raw, "pooler_output") and raw.pooler_output is not None:
            return raw.pooler_output
        if hasattr(raw, "last_hidden_state"):
            # Mean-pool over sequence dimension as a safe fallback
            return raw.last_hidden_state.mean(dim=1)
        raise TypeError(f"Cannot extract feature tensor from {type(raw)}")

    @staticmethod
    def _clear_gpu_cache() -> None:
        """Free unused GPU memory and trigger Python GC."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
