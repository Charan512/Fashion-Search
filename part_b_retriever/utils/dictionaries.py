"""
Fashion Retrieval Dictionaries.

Comprehensive vocabulary mappings for color names, clothing items,
setting locations, style categories, and formality keywords.
These are used by QueryDecomposer to parse natural language queries.
"""
from __future__ import annotations

from typing import Dict, List, Set

# ── Color Dictionary ──────────────────────────────────────────────────────────
# Maps canonical color names to all their synonyms/variants

COLOR_DICTIONARY: Dict[str, List[str]] = {
    "red": ["red", "crimson", "scarlet", "maroon", "ruby", "cherry", "burgundy", "rose"],
    "blue": ["blue", "navy", "navy blue", "cobalt", "royal blue", "azure", "sapphire", "indigo"],
    "green": ["green", "emerald", "olive", "forest green", "lime green", "sage", "mint"],
    "yellow": ["yellow", "golden", "mustard", "lemon", "amber", "canary", "blonde"],
    "orange": ["orange", "tangerine", "peach", "coral", "rust"],
    "purple": ["purple", "violet", "lavender", "lilac", "mauve", "plum"],
    "pink": ["pink", "hot pink", "blush", "fuchsia", "magenta", "salmon"],
    "brown": ["brown", "tan", "khaki", "caramel", "chocolate", "beige", "ivory"],
    "black": ["black", "jet black", "ebony"],
    "white": ["white", "off-white", "cream", "ivory", "snow", "pearl"],
    "gray": ["gray", "grey", "silver", "ash", "slate", "charcoal"],
    "teal": ["teal", "cyan", "turquoise", "aqua", "seafoam"],
    "gold": ["gold", "golden", "metallic gold"],
    "silver": ["silver", "metallic silver", "chrome"],
    "beige": ["beige", "nude", "sand", "buff", "ecru"],
    "lime": ["lime", "lime green", "chartreuse", "neon green"],
}

# Flat set of all color synonym strings (for quick membership test)
ALL_COLOR_SYNONYMS: Set[str] = {
    syn for synonyms in COLOR_DICTIONARY.values() for syn in synonyms
}


def get_canonical_color(synonym: str) -> str | None:
    """Return canonical color name for a synonym, or None if not found.

    Args:
        synonym: Color string to look up.

    Returns:
        Canonical color name (e.g., ``"red"`` for ``"crimson"``),
        or ``None`` if the synonym is not in the dictionary.
    """
    s = synonym.lower().strip()
    for canonical, synonyms in COLOR_DICTIONARY.items():
        if s in synonyms:
            return canonical
    return None


# ── Clothing Dictionary ───────────────────────────────────────────────────────
# Maps canonical clothing items to synonyms

CLOTHING_DICTIONARY: Dict[str, List[str]] = {
    "shirt": ["shirt", "button-down", "button-up", "dress shirt", "oxford"],
    "t-shirt": ["t-shirt", "tee", "tshirt", "t shirt", "polo"],
    "blouse": ["blouse", "top", "camisole"],
    "tie": ["tie", "necktie", "bow tie", "bowtie", "cravat"],
    "blazer": ["blazer", "sport coat", "sport jacket"],
    "jacket": ["jacket", "windbreaker", "bomber", "leather jacket"],
    "coat": ["coat", "overcoat", "trench coat", "peacoat", "parka"],
    "pants": ["pants", "trousers", "slacks", "chinos", "khakis"],
    "jeans": ["jeans", "denim", "denim pants"],
    "shorts": ["shorts", "bermuda shorts"],
    "skirt": ["skirt", "mini skirt", "midi skirt", "maxi skirt"],
    "dress": ["dress", "gown", "frock", "sundress"],
    "suit": ["suit", "two-piece suit", "three-piece suit", "business suit"],
    "sweater": ["sweater", "pullover", "knitwear", "jumper", "cardigan"],
    "hoodie": ["hoodie", "sweatshirt", "hooded sweatshirt"],
    "vest": ["vest", "waistcoat"],
    "scarf": ["scarf", "wrap", "muffler"],
    "hat": ["hat", "fedora", "beret", "beanie", "knit hat"],
    "cap": ["cap", "baseball cap", "snapback"],
    "shoes": ["shoes", "loafers", "oxfords", "moccasins", "flats"],
    "sneakers": ["sneakers", "trainers", "athletic shoes", "running shoes"],
    "boots": ["boots", "ankle boots", "combat boots", "knee-high boots"],
    "heels": ["heels", "high heels", "stilettos", "pumps", "wedges"],
    "raincoat": ["raincoat", "rain jacket", "waterproof jacket", "slicker", "mac"],
}

ALL_CLOTHING_SYNONYMS: Set[str] = {
    syn for synonyms in CLOTHING_DICTIONARY.values() for syn in synonyms
}


def get_canonical_clothing(synonym: str) -> str | None:
    """Return canonical clothing item for a synonym.

    Args:
        synonym: Clothing string to look up.

    Returns:
        Canonical item name or ``None``.
    """
    s = synonym.lower().strip()
    for canonical, synonyms in CLOTHING_DICTIONARY.items():
        if s in synonyms:
            return canonical
    return None


# ── Setting Dictionary ────────────────────────────────────────────────────────

SETTING_DICTIONARY: Dict[str, List[str]] = {
    "indoor_office": ["office", "workplace", "work", "boardroom", "conference room", "corporate"],
    "outdoor_park": ["park", "garden", "outdoor", "nature", "grass", "bench", "trees"],
    "outdoor_street": ["street", "city", "urban", "sidewalk", "pavement", "downtown", "city walk"],
    "home_interior": ["home", "house", "living room", "bedroom", "kitchen", "interior", "indoors"],
    "gym": ["gym", "fitness", "workout", "exercise", "athletic", "sports"],
    "beach": ["beach", "seaside", "ocean", "sea", "sand", "coastal"],
    "restaurant": ["restaurant", "cafe", "dining", "bar", "nightclub"],
    "shopping_mall": ["mall", "shopping center", "store", "boutique"],
}

ALL_SETTINGS: Set[str] = set(SETTING_DICTIONARY.keys())


def get_canonical_setting(text: str) -> str | None:
    """Find the setting that best matches a text snippet.

    Args:
        text: A word or phrase to match against settings.

    Returns:
        Canonical setting string or ``None``.
    """
    text_lower = text.lower()
    for canonical, keywords in SETTING_DICTIONARY.items():
        if any(kw in text_lower for kw in keywords):
            return canonical
    return None


# ── Style Dictionary ──────────────────────────────────────────────────────────

STYLE_DICTIONARY: Dict[str, List[str]] = {
    "business_formal": [
        "formal", "professional", "business formal", "business attire",
        "office wear", "corporate", "executive",
    ],
    "smart_casual": [
        "smart casual", "business casual", "semi-formal",
    ],
    "casual": [
        "casual", "relaxed", "everyday", "weekend", "laid-back",
        "comfortable", "simple",
    ],
    "athletic": [
        "athletic", "sporty", "fitness", "gym", "activewear",
        "workout", "running",
    ],
    "elegant": [
        "elegant", "sophisticated", "classy", "chic", "refined",
        "luxury", "upscale",
    ],
    "streetwear": [
        "streetwear", "street style", "urban", "hypebeast",
        "skater", "grunge",
    ],
    "bohemian": [
        "bohemian", "boho", "hippie", "festival", "artistic",
    ],
}

ALL_STYLES: Set[str] = set(STYLE_DICTIONARY.keys())


def get_canonical_style(text: str) -> str | None:
    """Find the style category matching the text.

    Args:
        text: Query snippet to match.

    Returns:
        Style category string or ``None``.
    """
    text_lower = text.lower()
    for canonical, keywords in STYLE_DICTIONARY.items():
        if any(kw in text_lower for kw in keywords):
            return canonical
    return None


# ── Formality Keywords ────────────────────────────────────────────────────────

FORMALITY_HIGH_KEYWORDS: List[str] = [
    "formal", "professional", "business", "office", "corporate",
    "executive", "suit", "tie", "blazer", "dress shirt",
]

FORMALITY_LOW_KEYWORDS: List[str] = [
    "casual", "relaxed", "comfortable", "weekend", "everyday",
    "jeans", "t-shirt", "sneakers", "hoodie", "athletic",
]


def estimate_formality_from_text(text: str) -> float:
    """Estimate formality level from query text alone.

    Args:
        text: Natural language query.

    Returns:
        Float in ``[0, 1]`` — 1.0 is very formal.
    """
    text_lower = text.lower()
    formal_hits = sum(1 for kw in FORMALITY_HIGH_KEYWORDS if kw in text_lower)
    casual_hits = sum(1 for kw in FORMALITY_LOW_KEYWORDS if kw in text_lower)

    total = formal_hits + casual_hits
    if total == 0:
        return 0.5  # neutral
    return round(formal_hits / total, 3)
