"""
Result Card component for the Fashion Retrieval demo.

Renders a single search result with image, score breakdown,
matching attributes, and a human-readable explanation.
"""
from __future__ import annotations

import json
import base64
import io
from pathlib import Path
from typing import Optional

import streamlit as st

from demo.components.theme import (
    ACCENT_GOLD,
    ERROR_RED,
    PRIMARY_GREEN,
    TEXT_MUTED,
)

# ── Thumbnail cache ───────────────────────────────────────────────────────────
# Pre-generated 320px JPEG thumbnails for all 1000 indexed images.
# Built by running: python3 scripts/build_thumbnails.py
# The COCO CDN only hosts a subset of Fashionpedia images publicly (most 404),
# so we embed thumbnails directly rather than relying on external URLs.
_THUMB_PATH = Path(__file__).parent.parent / "thumbnails.json"


@st.cache_data(show_spinner=False)
def _load_thumbnails() -> dict:
    """Load thumbnail base64 map. Cached once per Streamlit session."""
    if _THUMB_PATH.exists():
        with open(_THUMB_PATH) as f:
            return json.load(f)
    return {}


def _resolve_image(result):
    """Return a PIL image or HTTP URL for the result, or None if unavailable.

    Priority:
      1. ``result.image_url`` — explicit HTTP URL stored during indexing.
      2. Base64 thumbnail from local cache (thumbnails.json).
      3. None — caller renders the placeholder.
    """
    if result.image_url and result.image_url.startswith("http"):
        return result.image_url

    thumbs = _load_thumbnails()
    row_id = str(result.image_id).strip()
    b64 = thumbs.get(row_id)
    if b64:
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    return None



def render_result_card(
    rank: int,
    result,
    show_breakdown: bool = True,
) -> None:
    """Render a single fashion search result card.

    Args:
        rank: 1-based result rank (displayed as #1, #2, …).
        result: ``SearchResult`` object from the retriever.
        show_breakdown: Whether to show the score breakdown expander.
    """
    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        col_img, col_info = st.columns([1, 2], gap="medium")

        # ── Image column ──────────────────────────────────────
        with col_img:
            image = _resolve_image(result)
            if image is not None:
                try:
                    st.image(
                        image,
                        width="stretch",
                        caption=f"#{rank}",
                    )
                except Exception:
                    _render_placeholder(rank)
            else:
                _render_placeholder(rank)

        # ── Info column ───────────────────────────────────────
        with col_info:
            _render_rank_header(rank, result.overall_score)
            _render_score_bar(result.overall_score)
            _render_explanation(result.explanation)
            _render_attributes(result.matching_attributes, result.non_matching_attributes)

            if show_breakdown:
                _render_score_breakdown(result)

        st.markdown("</div>", unsafe_allow_html=True)


# ── Sub-components ────────────────────────────────────────────────────────────


def _render_rank_header(rank: int, score: float) -> None:
    """Render result rank and overall score."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <span style="
                background:{ACCENT_GOLD};
                color:#000;
                font-weight:800;
                font-size:1.1rem;
                padding:4px 12px;
                border-radius:20px;
            ">#{rank}</span>
            <span style="
                color:{PRIMARY_GREEN};
                font-size:1.4rem;
                font-weight:700;
            ">{score:.0%} match</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_score_bar(score: float) -> None:
    """Render a horizontal score bar."""
    pct = int(score * 100)
    st.markdown(
        f"""
        <div class="score-bar-container">
            <div class="score-bar-fill" style="width:{pct}%"></div>
        </div>
        <div style="color:{TEXT_MUTED};font-size:0.75rem;margin-bottom:10px;">
            Relevance score
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_explanation(explanation: str) -> None:
    """Render the human-readable explanation text."""
    if explanation:
        st.markdown(
            f'<p style="color:{TEXT_MUTED};font-size:0.85rem;margin-bottom:10px;">'
            f"{explanation}</p>",
            unsafe_allow_html=True,
        )


def _render_attributes(matching: list, non_matching: list) -> None:
    """Render matched and unmatched attribute badges."""
    if matching or non_matching:
        badges_html = ""
        for attr in matching:
            badges_html += f'<span class="attr-badge">✓ {attr}</span>'
        for attr in non_matching:
            badges_html += f'<span class="attr-badge attr-badge-miss">✗ {attr}</span>'
        st.markdown(
            f'<div style="margin-bottom:10px;">{badges_html}</div>',
            unsafe_allow_html=True,
        )


def _render_score_breakdown(result) -> None:
    """Render expandable score breakdown panel."""
    with st.expander("Score breakdown"):
        breakdown = result.score_breakdown
        cols = st.columns(3)
        labels = list(breakdown.keys())
        values = list(breakdown.values())

        for i, col in enumerate(cols):
            if i < len(labels):
                col.metric(labels[i], f"{values[i]:.3f}")


def _render_placeholder(rank: int) -> None:
    """Render a placeholder when no image URL is available."""
    st.markdown(
        f"""
        <div style="
            width:100%;
            aspect-ratio:1;
            background:linear-gradient(135deg, #1a1a1a, #2a2a2a);
            border-radius:8px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#555;
            font-size:1rem;
            letter-spacing:0.05em;
        ">No Image</div>
        <div style="text-align:center;color:#555;font-size:0.75rem;">#{rank}</div>
        """,
        unsafe_allow_html=True,
    )
