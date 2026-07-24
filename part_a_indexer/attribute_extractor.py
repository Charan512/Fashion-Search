"""
Attribute Extractor — Part A Indexer.

Predicts fine-grained fashion attributes from images using
zero-shot CLIP prompt engineering. No fine-tuning required.

Attributes extracted per image:
  - primary_colors / secondary_colors (top-3 from 18-color palette)
  - clothing_items (detected garment types with confidence)
  - formality_score (0.0 = casual, 1.0 = very formal)
  - setting (indoor_office, outdoor_park, street, home, gym, beach, …)
  - style_category (business_formal, casual, athletic, elegant, …)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Vocabulary constants ──────────────────────────────────────────────────────

COLOR_NAMES: List[str] = [
    "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "black", "white", "gray", "navy",
    "maroon", "teal", "lime", "gold", "silver", "beige",
]

CLOTHING_ITEMS: List[str] = [
    "shirt", "t-shirt", "blouse", "tie", "blazer", "jacket",
    "coat", "pants", "trousers", "jeans", "shorts", "skirt",
    "dress", "suit", "sweater", "hoodie", "vest", "scarf",
    "hat", "cap", "shoes", "sneakers", "boots", "heels", "raincoat",
]

SETTINGS: List[str] = [
    "indoor office", "outdoor park", "city street",
    "home interior", "gym", "beach", "restaurant", "shopping mall",
]

STYLE_CATEGORIES: List[str] = [
    "business formal", "smart casual", "casual", "athletic",
    "elegant", "streetwear", "bohemian",
]

FORMALITY_FORMAL_PROMPTS: List[str] = [
    "a formal outfit", "business attire", "professional clothing",
    "a suit and tie", "formal office wear",
]

FORMALITY_CASUAL_PROMPTS: List[str] = [
    "casual clothing", "a relaxed outfit", "everyday wear",
    "casual streetwear", "a t-shirt and jeans",
]


class AttributeExtractor:
    """Zero-shot attribute extraction using CLIP prompt engineering.

    Attributes are inferred by comparing image embeddings against
    textual prompts (e.g., "This clothing contains red") and picking
    the highest-scoring labels.

    Args:
        device: Compute device. Auto-detected if not specified.
        confidence_threshold: Minimum normalised score to retain
            a colour or clothing item in output lists.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence_threshold = confidence_threshold

        self._clip_model = None
        self._clip_preprocess = None
        self._text_cache: Dict[tuple, torch.Tensor] = {}

        logger.info("AttributeExtractor initialised (device=%s)", self.device)

    # ── Model access ──────────────────────────────────────────────────────────

    @property
    def clip_model(self):
        """CLIP model, loaded lazily."""
        if self._clip_model is None:
            self._load_clip()
        return self._clip_model

    @property
    def clip_preprocess(self):
        """CLIP preprocessor, loaded lazily."""
        if self._clip_preprocess is None:
            self._load_clip()
        return self._clip_preprocess

    def _load_clip(self) -> None:
        import clip

        logger.info("AttributeExtractor: loading CLIP (ViT-B/32)…")
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_colors(self, image: Optional[Image.Image] = None, image_features: Optional[torch.Tensor] = None) -> List[Tuple[str, float]]:
        """Predict clothing colors present in the image."""
        prompts = [f"This clothing contains the color {c}" for c in COLOR_NAMES]
        scores = self._score_prompts_fast(image, image_features, prompts)
        return self._top_labels(COLOR_NAMES, scores, top_k=3)

    def extract_clothing_items(self, image: Optional[Image.Image] = None, image_features: Optional[torch.Tensor] = None) -> List[Tuple[str, float]]:
        """Detect clothing items present in the image."""
        prompts = [f"This person is wearing a {item}" for item in CLOTHING_ITEMS]
        scores = self._score_prompts_fast(image, image_features, prompts)
        return self._top_labels(CLOTHING_ITEMS, scores, top_k=5)

    def score_formality(self, image: Optional[Image.Image] = None, image_features: Optional[torch.Tensor] = None) -> float:
        """Predict how formal the outfit in the image is."""
        all_prompts = FORMALITY_FORMAL_PROMPTS + FORMALITY_CASUAL_PROMPTS
        scores = self._score_prompts_fast(image, image_features, all_prompts)

        n_formal = len(FORMALITY_FORMAL_PROMPTS)
        formal_score = float(scores[:n_formal].mean())
        casual_score = float(scores[n_formal:].mean())

        # Normalise to [0, 1]
        total = formal_score + casual_score + 1e-10
        return round(formal_score / total, 4)

    def classify_setting(self, image: Optional[Image.Image] = None, image_features: Optional[torch.Tensor] = None) -> str:
        """Classify the location/setting of the image."""
        prompts = [f"A photo taken at {s}" for s in SETTINGS]
        scores = self._score_prompts_fast(image, image_features, prompts)
        best_idx = int(np.argmax(scores))
        raw = SETTINGS[best_idx]
        return raw.replace(" ", "_")

    def classify_style(self, image: Optional[Image.Image] = None, image_features: Optional[torch.Tensor] = None) -> str:
        """Classify the overall fashion style of the image."""
        prompts = [f"This is a {s} fashion look" for s in STYLE_CATEGORIES]
        scores = self._score_prompts_fast(image, image_features, prompts)
        best_idx = int(np.argmax(scores))
        return STYLE_CATEGORIES[best_idx].replace(" ", "_")

    def extract_all_attributes(self, image: Image.Image) -> Dict:
        """Extract all attributes from a single image optimally."""
        # Calculate image features exactly ONCE per image.
        image_features = self._get_image_features(image)

        colors = self.extract_colors(image_features=image_features)
        clothing = self.extract_clothing_items(image_features=image_features)
        formality = self.score_formality(image_features=image_features)
        setting = self.classify_setting(image_features=image_features)
        style = self.classify_style(image_features=image_features)

        color_names = [c for c, _ in colors]
        primary_colors = color_names[:2]
        secondary_colors = color_names[2:]

        return {
            "primary_colors": primary_colors,
            "secondary_colors": secondary_colors,
            "clothing_items": [item for item, _ in clothing],
            "formality_score": formality,
            "setting": setting,
            "style_category": style,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_image_features(self, image: Image.Image) -> torch.Tensor:
        """Compute and normalise image features from the vision transformer."""
        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _get_text_features(self, prompts: List[str]) -> torch.Tensor:
        """Compute and cache text features from the text transformer."""
        cache_key = tuple(prompts)
        if cache_key not in self._text_cache:
            import clip
            with torch.no_grad():
                text_tokens = clip.tokenize(prompts, truncate=True).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self._text_cache[cache_key] = text_features
        return self._text_cache[cache_key]

    def _score_prompts_fast(
        self, image: Optional[Image.Image], image_features: Optional[torch.Tensor], prompts: List[str]
    ) -> np.ndarray:
        """Compute normalised CLIP similarity scores extremely efficiently."""
        if image_features is None:
            if image is None:
                raise ValueError("Either image or image_features must be provided")
            image_features = self._get_image_features(image)
            
        text_features = self._get_text_features(prompts)
        
        with torch.no_grad():
            # Cosine similarity → softmax scores
            logits = (image_features @ text_features.T) * 100.0  # scale like CLIP paper
            probs = torch.softmax(logits, dim=-1)[0]

        return probs.cpu().float().numpy()

    def _score_prompts(self, image: Image.Image, prompts: List[str]) -> np.ndarray:
        """Legacy method for backward compatibility."""
        return self._score_prompts_fast(image, None, prompts)

    def _top_labels(
        self,
        labels: List[str],
        scores: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return the top-k (label, score) pairs above the threshold."""
        indices = np.argsort(scores)[::-1][:top_k]
        result = []
        for idx in indices:
            score = float(scores[idx])
            if score >= self.confidence_threshold:
                result.append((labels[idx], round(score, 4)))
        return result
