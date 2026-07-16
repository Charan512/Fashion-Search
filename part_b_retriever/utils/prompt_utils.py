"""
CLIP Prompt Templates — Part B Retriever.

Centralised prompt construction helpers used by the embedding
extractor and attribute matcher for consistent text encoding.
"""
from __future__ import annotations

from typing import List


def build_color_prompt(color: str) -> str:
    """Build a CLIP-friendly color classification prompt.

    Args:
        color: Color name (e.g., ``"red"``).

    Returns:
        Text prompt string.
    """
    return f"A photo of clothing in {color} color"


def build_clothing_prompt(item: str) -> str:
    """Build a CLIP-friendly clothing item prompt.

    Args:
        item: Clothing item name (e.g., ``"tie"``).

    Returns:
        Text prompt string.
    """
    return f"A photo of a person wearing a {item}"


def build_setting_prompt(setting: str) -> str:
    """Build a CLIP-friendly setting/location prompt.

    Args:
        setting: Setting name (e.g., ``"office"``).

    Returns:
        Text prompt string.
    """
    return f"A photo taken in a {setting.replace('_', ' ')}"


def build_style_prompt(style: str) -> str:
    """Build a CLIP-friendly fashion style prompt.

    Args:
        style: Style category (e.g., ``"business_formal"``).

    Returns:
        Text prompt string.
    """
    return f"A {style.replace('_', ' ')} fashion outfit"


def build_formality_prompts() -> List[str]:
    """Return formal and casual reference prompts for formality scoring.

    Returns:
        List of prompts — first half are formal, second half casual.
    """
    formal = [
        "A formal business outfit",
        "Professional workplace attire",
        "A suit and tie combination",
        "Executive office fashion",
        "A formal dress and heels",
    ]
    casual = [
        "A casual everyday outfit",
        "Relaxed weekend clothing",
        "Jeans and a t-shirt look",
        "Comfortable casual streetwear",
        "A laid-back simple outfit",
    ]
    return formal + casual


def build_compositional_query(colors: List[str], items: List[str], context: str = "") -> str:
    """Build a rich query string from decomposed components.

    Useful for creating improved CLIP text embeddings that capture
    the full compositionality of a query.

    Args:
        colors: List of color strings.
        items: List of clothing item strings.
        context: Optional context/setting string.

    Returns:
        Combined natural language query.
    """
    parts = []
    if colors and items:
        color_item_pairs = [f"{c} {i}" for c, i in zip(colors, items)]
        parts.append(" and ".join(color_item_pairs))
    elif colors:
        parts.append(" and ".join(colors) + " colored clothing")
    elif items:
        parts.append(" and ".join(items))

    if context:
        parts.append(f"in {context}")

    return " ".join(parts) if parts else "fashion clothing"
