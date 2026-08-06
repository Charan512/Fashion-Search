"""
Fashion Retrieval — Gradio App Entry Point.

Replaces the Streamlit multi-page app with a single Gradio Blocks
application containing three tabs: Search, Examples, and About.

Run with:
    python demo/gradio_app.py
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

# Ensure project root is on Python path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants (mirrors theme.py) ──────────────────────────────────────────────
PRIMARY_GREEN = "#2ECC71"
ACCENT_GOLD = "#F39C12"
DARK_BG = "#0d0d0d"
DARK_SECONDARY = "#1a1a1a"
DARK_CARD = "#222222"
TEXT_WHITE = "#F0F0F0"
TEXT_MUTED = "#A0A0A0"
ERROR_RED = "#C0392B"
BORDER_GOLD = "#B8860B"

EXAMPLE_QUERIES = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
]

EVALUATION_QUERIES = [
    {
        "query": "A person in a bright yellow raincoat",
        "title": "Query 1 — Attribute Specific",
        "description": (
            "Tests color classification and single-item focus. "
            "The system must return images with bright yellow outerwear, not orange or green."
        ),
        "tests": ["Color precision", "Single attribute", "Clothing type"],
    },
    {
        "query": "Professional business attire inside a modern office",
        "title": "Query 2 — Contextual / Place",
        "description": (
            "Tests context understanding and formality classification. "
            "Results must show formal office wear in a corporate setting."
        ),
        "tests": ["Setting detection", "Formality scoring", "Context match"],
    },
    {
        "query": "Someone wearing a blue shirt sitting on a park bench",
        "title": "Query 3 — Complex Semantic",
        "description": (
            "Tests multi-attribute + pose + location awareness. "
            "Combines color, clothing, setting, and position."
        ),
        "tests": ["Multi-attribute", "Location (park)", "Color + item"],
    },
    {
        "query": "Casual weekend outfit for a city walk",
        "title": "Query 4 — Style Inference",
        "description": (
            "Tests style classification without explicit item names. "
            "The system must infer 'casual' from semantic context."
        ),
        "tests": ["Style inference", "Implicit attributes", "Urban context"],
    },
    {
        "query": "A red tie and a white shirt in a formal setting",
        "title": "Query 5 — Compositional (CORE)",
        "description": (
            "Core compositionality test. Must distinguish 'red tie + white shirt' "
            "from 'white tie + red shirt'. This is where vanilla CLIP fails."
        ),
        "tests": ["Color-item composition", "Formality", "Multi-garment"],
    },
]

# ── Thumbnail cache ───────────────────────────────────────────────────────────
_THUMB_PATH = Path(__file__).parent / "thumbnails.json"
_THUMBNAILS: Optional[Dict[str, str]] = None


def _get_thumbnails() -> Dict[str, str]:
    """Load & cache thumbnail base64 map once."""
    global _THUMBNAILS
    if _THUMBNAILS is None:
        if _THUMB_PATH.exists():
            logger.info("Loading thumbnails.json …")
            with open(_THUMB_PATH) as f:
                _THUMBNAILS = json.load(f)
            logger.info("Loaded %d thumbnails.", len(_THUMBNAILS))
        else:
            _THUMBNAILS = {}
    return _THUMBNAILS


def _resolve_image(result) -> Optional[Any]:
    """Return a PIL Image (from local cache) or an HTTP URL string, or None."""
    if result.image_url and result.image_url.startswith("http"):
        return result.image_url

    thumbs = _get_thumbnails()
    row_id = str(result.image_id).strip()
    b64 = thumbs.get(row_id)
    if b64:
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    return None


# ── Retriever (loaded once on startup) ────────────────────────────────────────
_retriever = None


def _get_retriever():
    """Load FashionRetriever once and reuse."""
    global _retriever
    if _retriever is None:
        from part_b_retriever.retriever import FashionRetriever
        _retriever = FashionRetriever(device=os.environ.get("DEVICE", "cpu"))
    return _retriever


# ── HTML helpers ──────────────────────────────────────────────────────────────

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {{
    --primary-green: {PRIMARY_GREEN};
    --accent-gold:   {ACCENT_GOLD};
    --dark-bg:       {DARK_BG};
    --dark-secondary:{DARK_SECONDARY};
    --dark-card:     {DARK_CARD};
    --text-white:    {TEXT_WHITE};
    --text-muted:    {TEXT_MUTED};
    --border-gold:   {BORDER_GOLD};
}}

body, .gradio-container, .gr-block {{
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, {DARK_BG} 0%, #111111 100%) !important;
    color: {TEXT_WHITE} !important;
}}

/* Headings */
h1, h2, h3 {{
    color: {PRIMARY_GREEN} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}}

h4, h5, h6 {{
    color: {ACCENT_GOLD} !important;
    font-weight: 600 !important;
}}

/* Tabs */
.tab-nav button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: {TEXT_MUTED} !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}}

.tab-nav button.selected {{
    color: {PRIMARY_GREEN} !important;
    border-bottom: 2px solid {PRIMARY_GREEN} !important;
}}

/* Inputs */
textarea, input[type="text"], input[type="number"] {{
    background-color: {DARK_CARD} !important;
    color: {TEXT_WHITE} !important;
    border: 1px solid {BORDER_GOLD} !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}}

textarea:focus, input:focus {{
    border-color: {PRIMARY_GREEN} !important;
    box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.2) !important;
}}

/* Buttons */
.gr-button-primary, button.primary {{
    background: linear-gradient(135deg, {PRIMARY_GREEN}, #27AE60) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(46,204,113,0.3) !important;
}}

.gr-button-primary:hover, button.primary:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(46,204,113,0.5) !important;
}}

button.secondary {{
    background: {DARK_CARD} !important;
    color: {ACCENT_GOLD} !important;
    border: 1px solid {BORDER_GOLD} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}

button.secondary:hover {{
    border-color: {PRIMARY_GREEN} !important;
    color: {PRIMARY_GREEN} !important;
}}

/* Result cards */
.result-card {{
    background: {DARK_CARD};
    border: 1px solid {BORDER_GOLD};
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}}

.result-card:hover {{
    border-color: {PRIMARY_GREEN};
    box-shadow: 0 4px 20px rgba(46, 204, 113, 0.15);
    transform: translateY(-2px);
}}

/* Score bars */
.score-bar-container {{
    background: #333;
    border-radius: 4px;
    height: 6px;
    margin: 4px 0 8px;
}}

.score-bar-fill {{
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, {PRIMARY_GREEN}, {ACCENT_GOLD});
    transition: width 0.4s ease;
}}

/* Attribute badges */
.attr-badge {{
    display: inline-block;
    background: rgba(46, 204, 113, 0.15);
    border: 1px solid {PRIMARY_GREEN};
    color: {PRIMARY_GREEN};
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}}

.attr-badge-miss {{
    background: rgba(192, 57, 43, 0.15);
    border: 1px solid {ERROR_RED};
    color: {ERROR_RED};
}}

/* Hero gradient panel */
.search-hero {{
    background: linear-gradient(135deg,
        rgba(46,204,113,0.05) 0%,
        rgba(243,156,18,0.05) 100%
    );
    border: 1px solid rgba(243,156,18,0.2);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}}

/* HR */
hr {{
    border-color: {BORDER_GOLD} !important;
    opacity: 0.4;
}}

/* Sliders */
input[type=range] {{
    accent-color: {ACCENT_GOLD};
}}

/* Accordion / details */
details summary {{
    color: {ACCENT_GOLD} !important;
    font-weight: 600 !important;
}}

/* Example query chips */
.example-chip {{
    display: inline-block;
    background: {DARK_CARD};
    border: 1px solid {BORDER_GOLD};
    color: {ACCENT_GOLD};
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px;
    cursor: pointer;
    transition: all 0.2s ease;
}}

.example-chip:hover {{
    border-color: {PRIMARY_GREEN};
    color: {PRIMARY_GREEN};
    background: rgba(46,204,113,0.08);
}}

/* Metric cards */
.metric-card {{
    background: {DARK_SECONDARY};
    border: 1px solid {BORDER_GOLD};
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}}

.metric-value {{
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}}

.metric-label {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
}}

/* No results / error states */
.no-results {{
    text-align: center;
    padding: 3rem;
    color: {TEXT_MUTED};
    font-size: 1rem;
    background: {DARK_CARD};
    border-radius: 12px;
    border: 1px dashed {BORDER_GOLD};
}}

/* Query decomposition debug box */
.decomp-card {{
    background: {DARK_SECONDARY};
    border: 1px solid {BORDER_GOLD};
    border-radius: 10px;
    padding: 1rem;
    margin-top: 0.5rem;
}}

/* Hero header */
.hero-header {{
    text-align: center;
    padding: 2rem 1rem 1rem;
}}

.hero-header h1 {{
    font-size: 2.6rem !important;
    margin-bottom: 0.5rem;
}}

.hero-header p {{
    color: {TEXT_MUTED};
    font-size: 1rem;
    max-width: 600px;
    margin: 0 auto;
}}
"""


# ── HTML builders ─────────────────────────────────────────────────────────────

def _image_to_data_uri(result) -> str:
    """Return a data URI string or an empty string if no image."""
    img = _resolve_image(result)
    if img is None:
        return ""
    if isinstance(img, str):
        return img  # HTTP URL
    # PIL Image → data URI
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _render_result_card_html(rank: int, result) -> str:
    """Build an HTML string for a single search result card."""
    score_pct = int(result.overall_score * 100)
    img_src = _image_to_data_uri(result)

    if img_src:
        img_html = (
            f'<img src="{img_src}" '
            f'style="width:100%;border-radius:8px;object-fit:cover;max-height:280px;" '
            f'alt="Result #{rank}" loading="lazy">'
        )
    else:
        img_html = (
            '<div style="width:100%;height:200px;background:linear-gradient(135deg,#1a1a1a,#2a2a2a);'
            'border-radius:8px;display:flex;align-items:center;justify-content:center;'
            f'color:#555;font-size:0.9rem;">No Image</div>'
        )

    # Attribute badges
    match_badges = "".join(
        f'<span class="attr-badge">✓ {a}</span>' for a in result.matching_attributes
    )
    miss_badges = "".join(
        f'<span class="attr-badge attr-badge-miss">✗ {a}</span>'
        for a in result.non_matching_attributes
    )
    badges_html = match_badges + miss_badges

    # Score breakdown rows
    bd = result.score_breakdown
    breakdown_rows = "".join(
        f'<tr><td style="color:{TEXT_MUTED};font-size:0.8rem;padding:2px 8px 2px 0;">{k}</td>'
        f'<td style="color:{PRIMARY_GREEN};font-weight:600;font-size:0.8rem;">{v:.3f}</td></tr>'
        for k, v in bd.items()
    )
    breakdown_html = (
        f'<details style="margin-top:8px;">'
        f'<summary style="font-size:0.82rem;color:{ACCENT_GOLD};cursor:pointer;font-weight:600;">'
        f'Score breakdown</summary>'
        f'<table style="margin-top:6px;border-collapse:collapse;">{breakdown_rows}</table>'
        f'</details>'
    )

    explanation_html = ""
    if result.explanation:
        explanation_html = (
            f'<p style="color:{TEXT_MUTED};font-size:0.83rem;margin:6px 0 8px;">'
            f'{result.explanation}</p>'
        )

    return f"""
    <div class="result-card">
      <div style="display:flex;gap:16px;align-items:flex-start;">
        <!-- Image column -->
        <div style="flex:0 0 160px;max-width:160px;">
          {img_html}
        </div>
        <!-- Info column -->
        <div style="flex:1;min-width:0;">
          <!-- Rank + score -->
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="background:{ACCENT_GOLD};color:#000;font-weight:800;font-size:0.95rem;
                         padding:3px 10px;border-radius:20px;">#{rank}</span>
            <span style="color:{PRIMARY_GREEN};font-size:1.2rem;font-weight:700;">{score_pct}% match</span>
          </div>
          <!-- Score bar -->
          <div class="score-bar-container">
            <div class="score-bar-fill" style="width:{score_pct}%;"></div>
          </div>
          <div style="color:{TEXT_MUTED};font-size:0.72rem;margin-bottom:6px;">Relevance score</div>
          {explanation_html}
          <!-- Attribute badges -->
          <div style="margin-bottom:6px;">{badges_html}</div>
          {breakdown_html}
        </div>
      </div>
    </div>
    """


def _build_results_html(results: List, elapsed_ms: float) -> str:
    """Build the full results panel HTML."""
    count = len(results)
    header = (
        f'<div style="margin-bottom:1rem;">'
        f'<span style="color:{PRIMARY_GREEN};font-weight:700;font-size:1.15rem;">'
        f'{count} result{"s" if count != 1 else ""} found</span>'
        f'<span style="color:#666;font-size:0.82rem;margin-left:0.8rem;">in {elapsed_ms:.0f}ms</span>'
        f'</div>'
    )
    cards = "".join(_render_result_card_html(i + 1, r) for i, r in enumerate(results))
    return header + cards


def _build_decomp_html(components: Dict) -> str:
    """Build query decomposition debug HTML."""
    colors = components.get("colors", [])
    clothing = components.get("clothing", [])
    ctx = components.get("context", {})
    style = components.get("style", "—")

    def _color_row(c: dict) -> str:
        item_sfx = f" → {c['item']}" if c.get("item") else ""
        return (
            f'<li style="color:{TEXT_MUTED};font-size:0.82rem;">'
            f'<code style="color:{PRIMARY_GREEN};">{c["color"]}</code>'
            f'{item_sfx}'
            f' ({c["confidence"]:.0%})</li>'
        )
    color_rows = "".join(_color_row(c) for c in colors) or \
        f'<li style="color:#555;font-size:0.82rem;">None detected</li>'

    clothing_rows = "".join(
        f'<li style="color:{TEXT_MUTED};font-size:0.82rem;">'
        f'<code style="color:{PRIMARY_GREEN};">{cl["item"]}</code>'
        f' ({cl["confidence"]:.0%})</li>'
        for cl in clothing
    ) or f'<li style="color:#555;font-size:0.82rem;">None detected</li>'

    return f"""
    <div class="decomp-card">
      <h4 style="color:{ACCENT_GOLD};margin-top:0;font-size:0.95rem;">🔬 Query Decomposition</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
        <div>
          <p style="font-weight:600;color:{TEXT_WHITE};font-size:0.85rem;margin-bottom:4px;">
            Colors ({len(colors)})
          </p>
          <ul style="margin:0;padding-left:14px;">{color_rows}</ul>
        </div>
        <div>
          <p style="font-weight:600;color:{TEXT_WHITE};font-size:0.85rem;margin-bottom:4px;">
            Clothing ({len(clothing)})
          </p>
          <ul style="margin:0;padding-left:14px;">{clothing_rows}</ul>
        </div>
        <div>
          <p style="font-weight:600;color:{TEXT_WHITE};font-size:0.85rem;margin-bottom:4px;">Context</p>
          <ul style="margin:0;padding-left:14px;list-style:none;">
            <li style="color:{TEXT_MUTED};font-size:0.82rem;">
              Setting: <code style="color:{PRIMARY_GREEN};">{ctx.get("setting","—")}</code>
            </li>
            <li style="color:{TEXT_MUTED};font-size:0.82rem;">
              Formality: <code style="color:{PRIMARY_GREEN};">{ctx.get("formality",0.5):.0%}</code>
            </li>
            <li style="color:{TEXT_MUTED};font-size:0.82rem;">
              Style: <code style="color:{PRIMARY_GREEN};">{style}</code>
            </li>
          </ul>
        </div>
      </div>
    </div>
    """


# ── Search function (called by Gradio event) ──────────────────────────────────

def run_search(query: str, top_k: int, show_decomp: bool):
    """Main search handler wired to the Gradio submit event.

    Returns:
        (results_html, decomp_html)
    """
    query = (query or "").strip()
    if not query:
        return (
            f'<div class="no-results">⚠️ Please enter a search query.</div>',
            "",
        )

    try:
        retriever = _get_retriever()
    except Exception as exc:
        return (
            f'<div class="no-results" style="color:{ERROR_RED};">'
            f'❌ Failed to load retriever: {exc}<br>'
            f'Make sure <code>PINECONE_API_KEY</code> is set and the index is populated.'
            f'</div>',
            "",
        )

    decomp_html = ""
    if show_decomp:
        try:
            components = retriever.get_query_components(query)
            decomp_html = _build_decomp_html(components)
        except Exception as exc:
            decomp_html = f'<p style="color:{ERROR_RED};">Could not decompose query: {exc}</p>'

    t0 = time.time()
    try:
        results = retriever.search(query, top_k=int(top_k))
    except Exception as exc:
        return (
            f'<div class="no-results" style="color:{ERROR_RED};">❌ Search failed: {exc}</div>',
            decomp_html,
        )

    elapsed_ms = (time.time() - t0) * 1000

    if not results:
        return (
            f'<div class="no-results">'
            f'🔍 No results found for <em>"{query}"</em>. Try a different query.'
            f'</div>',
            decomp_html,
        )

    return _build_results_html(results, elapsed_ms), decomp_html


# ── Static page HTML ──────────────────────────────────────────────────────────

def _examples_tab_html() -> str:
    cards = []
    for i, eq in enumerate(EVALUATION_QUERIES):
        is_core = i == 4
        badges = "".join(f'<span class="attr-badge">✓ {t}</span>' for t in eq["tests"])
        border = ACCENT_GOLD if is_core else PRIMARY_GREEN
        core_tag = (
            f'<span style="color:{ERROR_RED};font-size:0.72rem;font-weight:700;'
            f'margin-left:8px;border:1px solid {ERROR_RED};border-radius:4px;'
            f'padding:1px 5px;">CORE TEST</span>'
            if is_core else ""
        )
        cards.append(f"""
        <div class="result-card">
          <h4 style="color:{ACCENT_GOLD};margin-top:0;">{eq["title"]}{core_tag}</h4>
          <p style="color:{TEXT_MUTED};font-size:0.83rem;margin-bottom:8px;">{eq["description"]}</p>
          <div style="background:#111;border-left:3px solid {border};
                      padding:8px 14px;border-radius:4px;font-family:monospace;
                      color:{border};margin-bottom:10px;font-size:0.88rem;">
            "{eq["query"]}"
          </div>
          <div>{badges}</div>
        </div>
        """)
    return "".join(cards)


def _about_tab_html() -> str:
    innovations = [
        ("Compositional Color-Item Binding",
         "Associates 'red' with 'tie' and 'white' with 'shirt' — not just 'red' and 'white' in a bag-of-words way."),
        ("Context-Aware Retrieval",
         "Detects location (office, park, beach) and formality levels from the query text."),
        ("Multi-Vector Search",
         "Parallel Pinecone queries with CLIP and FashionCLIP, merged with union scoring."),
        ("Hard Constraint Filtering",
         "Very formal queries enforce a formality_score threshold — casual images are filtered out."),
        ("Diversity Re-ranking",
         "Greedy MMR-like diversification prevents consecutive same-style results."),
        ("Explainability",
         "Every result includes a score breakdown and human-readable explanation."),
    ]
    inno_cards = "".join(
        f'<div class="result-card" style="margin-bottom:0.6rem;">'
        f'<strong style="color:{PRIMARY_GREEN};">{t}</strong>'
        f'<p style="color:{TEXT_MUTED};font-size:0.83rem;margin:4px 0 0;">{d}</p>'
        f'</div>'
        for t, d in innovations
    )

    queries = [
        ("A person in a bright yellow raincoat", False),
        ("Professional business attire inside a modern office", False),
        ("Someone wearing a blue shirt sitting on a park bench", False),
        ("Casual weekend outfit for a city walk", False),
        ("A red tie and a white shirt in a formal setting", True),
    ]
    def _query_row(i: int, q: str, is_core: bool) -> str:
        core_badge = f"<span style='color:{ERROR_RED};font-size:0.72rem;font-weight:700;'>CORE</span>" if is_core else ""
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;'
            f'margin-bottom:6px;background:#1a1a1a;border-radius:8px;'
            f'border-left:3px solid {ACCENT_GOLD if is_core else PRIMARY_GREEN};">'
            f'<span style="color:{ACCENT_GOLD};font-weight:700;min-width:20px;">#{i}</span>'
            f'<span style="font-family:monospace;color:#ccc;font-size:0.88rem;">{q}</span>'
            f'{core_badge}</div>'
        )
    
    query_rows = "".join(_query_row(i, q, is_core) for i, (q, is_core) in enumerate(queries, 1))

    scoring_cols = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:2rem;">
      <div class="metric-card">
        <div class="metric-value" style="color:{PRIMARY_GREEN};">50%</div>
        <div style="font-weight:600;margin-bottom:4px;">Semantic (CLIP)</div>
        <div class="metric-label">Global scene + context understanding. ViT-B/32 cosine similarity.</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:{ACCENT_GOLD};">30%</div>
        <div style="font-weight:600;margin-bottom:4px;">Fashion (FashionCLIP)</div>
        <div class="metric-label">Fashion-domain fine-tuned encoder. Better at clothing recognition.</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:{ERROR_RED};">20%</div>
        <div style="font-weight:600;margin-bottom:4px;">Attributes</div>
        <div class="metric-label">Explicit attribute matching: color, item, setting, formality.</div>
      </div>
    </div>
    """

    arch_diagram = f"""
    <div class="result-card">
      <pre style="color:#ccc;font-size:0.8rem;background:transparent;border:none;overflow-x:auto;">
USER QUERY (natural language)
        │
        ▼
┌──────────────────────┐
│   QueryDecomposer    │  ← dict-based, no ML required
│  colors / clothing   │
│  setting / formality │
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────┐
│  CLIP   │  │  FashionCLIP │
│  (50%)  │  │    (30%)     │
└────┬────┘  └──────┬───────┘
     │              │
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │    Pinecone   │  ← 512-D cosine search
    │  Vector Index │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Attr Matcher │  ← metadata scoring (20%)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ ResultRanker  │  ← hard constraints + diversity
    └───────┬───────┘
            │
            ▼
    SearchResult[]  ← with explanation + score breakdown
      </pre>
    </div>
    """

    return f"""
    <h2 style="color:{PRIMARY_GREEN};">Architecture</h2>
    {arch_diagram}
    <hr/>
    <h2 style="color:{PRIMARY_GREEN};">Scoring Weights</h2>
    {scoring_cols}
    <hr/>
    <h2 style="color:{PRIMARY_GREEN};">Key Innovations</h2>
    {inno_cards}
    <hr/>
    <h2 style="color:{PRIMARY_GREEN};">Evaluation Queries</h2>
    {query_rows}
    """


# ── Gradio Blocks app ─────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Fashion Retrieval System",
    ) as demo:

        # ── Hero header ───────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="hero-header">
          <h1>Fashion Retrieval System</h1>
          <p>
            Multi-vector fashion image search using CLIP + FashionCLIP + zero-shot attribute decomposition.
            Find fashion images using natural language descriptions.
          </p>
        </div>
        """)

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Search tab ────────────────────────────────────────────────────
            with gr.TabItem("🔍 Search"):

                # Quick-example chips (rendered as HTML buttons via JS)
                def _build_chip(q: str) -> str:
                    q_short = q[:35] + ("…" if len(q) > 35 else "")
                    # No backslashes in f-string, use double quotes for HTML and single for JS
                    onclick_js = f"document.getElementById('query-input').querySelector('textarea').value='{q}';document.getElementById('chips-row').dataset.chip='{q}';"
                    return f'<span class="example-chip" onclick="{onclick_js}">{q_short}</span>'

                chips_html = "".join(_build_chip(q) for q in EXAMPLE_QUERIES)

                gr.HTML(f"""
                <div style="margin-bottom:0.75rem;">
                  <p style="color:{ACCENT_GOLD};font-size:0.84rem;font-weight:600;margin-bottom:6px;">
                    Quick examples:
                  </p>
                  <div id="chips-row">
                    {chips_html}
                  </div>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=5):
                        query_input = gr.Textbox(
                            label="Describe what you're looking for",
                            placeholder="e.g., A red tie and white shirt in a formal office setting",
                            lines=3,
                            elem_id="query-input",
                        )
                    with gr.Column(scale=1, min_width=180):
                        top_k_slider = gr.Slider(
                            label="Number of results",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=10,
                        )

                with gr.Row():
                    search_btn = gr.Button("Search", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg")
                    decomp_toggle = gr.Checkbox(label="🔬 Show query decomposition", value=False)

                # Decomposition debug panel
                decomp_output = gr.HTML(visible=True)

                gr.HTML(f'<hr style="border-color:{BORDER_GOLD};opacity:0.3;margin:1rem 0;">')

                # Results area
                results_output = gr.HTML(
                    value=f'<div class="no-results" style="color:{TEXT_MUTED};">'
                          f'Enter a query above and press Search to find fashion images.'
                          f'</div>'
                )

                # Populate query from example chip click
                # (We use a JS-event-free approach: expose EXAMPLE_QUERIES as
                # clickable gr.Button rows for a11y and clean wiring)
                with gr.Row(visible=False):
                    chip_btns = [
                        gr.Button(q, elem_id=f"chip-btn-{i}")
                        for i, q in enumerate(EXAMPLE_QUERIES)
                    ]

                # Wire chip buttons → query input
                for btn, q in zip(chip_btns, EXAMPLE_QUERIES):
                    btn.click(fn=lambda x=q: x, outputs=query_input)

                # Wire search button
                search_btn.click(
                    fn=run_search,
                    inputs=[query_input, top_k_slider, decomp_toggle],
                    outputs=[results_output, decomp_output],
                )

                # Also allow submit on Enter (query_input submit)
                query_input.submit(
                    fn=run_search,
                    inputs=[query_input, top_k_slider, decomp_toggle],
                    outputs=[results_output, decomp_output],
                )

                # Clear button
                clear_btn.click(
                    fn=lambda: ("", "", ""),
                    outputs=[query_input, results_output, decomp_output],
                )

            # ── Examples tab ──────────────────────────────────────────────────
            with gr.TabItem("📋 Examples"):
                gr.HTML(f"""
                <div style="margin-bottom:1rem;">
                  <h2 style="color:{PRIMARY_GREEN};">Evaluation Examples</h2>
                  <p style="color:{TEXT_MUTED};">
                    These are the 5 official evaluation queries used to assess the system.
                    Each tests a different capability of the Fashion Attribute Pyramid.
                  </p>
                  <hr/>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML(_examples_tab_html())
                    with gr.Column(scale=1):
                        gr.HTML(f"""
                        <div class="result-card" style="text-align:center;">
                          <h4 style="color:{ACCENT_GOLD};">Run a Query</h4>
                          <p style="color:{TEXT_MUTED};font-size:0.85rem;">
                            Select a query and click a button below to run it in the Search tab.
                          </p>
                        </div>
                        """)
                        with gr.Column():
                            example_run_btns = []
                            for i, eq in enumerate(EVALUATION_QUERIES):
                                btn = gr.Button(
                                    f"▶ Run Query {i+1}",
                                    variant="secondary",
                                    elem_id=f"example-run-{i}",
                                )
                                example_run_btns.append((btn, eq["query"]))

            # ── About tab ─────────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.HTML(_about_tab_html())

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div style="text-align:center;padding:1.5rem 0 0.5rem;border-top:1px solid {BORDER_GOLD};
                    margin-top:2rem;color:{TEXT_MUTED};font-size:0.8rem;">
          Fashion Attribute Pyramid · CLIP + FashionCLIP · Pinecone · 44,200 images indexed
        </div>
        """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path=None,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="green",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ),
    )
