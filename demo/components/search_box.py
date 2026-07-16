"""
Search Box component for the Fashion Retrieval demo.
"""
from __future__ import annotations

from typing import Tuple

import streamlit as st

from demo.components.theme import ACCENT_GOLD, PRIMARY_GREEN

EXAMPLE_QUERIES = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
]


def render_search_box() -> Tuple[str, int, bool]:
    """Render the styled search input panel.

    Returns:
        ``(query, top_k, search_clicked)`` tuple.
    """
    st.markdown('<div class="search-hero">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:{ACCENT_GOLD};margin-bottom:0.5rem;">
            Describe what you're looking for
        </h3>
        <p style="color:#888;margin-bottom:1rem;font-size:0.9rem;">
            Use natural language — colors, clothing items, context, style.
        </p>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_area(
        label="Query",
        placeholder="e.g., A red tie and white shirt in a formal office",
        height=90,
        label_visibility="collapsed",
        key="search_query",
    )

    col_k, col_btn, col_clear = st.columns([2, 1, 1])

    with col_k:
        top_k = st.slider(
            "Results",
            min_value=1,
            max_value=20,
            value=10,
            key="top_k",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button(
            "Search",
            type="primary",
            use_container_width=True,
            key="search_button",
        )

    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True, key="clear_button"):
            st.session_state.pop("search_query", None)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return query, top_k, search_clicked


def render_example_chips() -> None:
    """Render clickable example query chips."""
    st.markdown(
        f'<p style="color:{ACCENT_GOLD};font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;">'
        "Try an example:",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(EXAMPLE_QUERIES))
    for i, (col, query) in enumerate(zip(cols, EXAMPLE_QUERIES)):
        with col:
            short = query[:30] + "…" if len(query) > 30 else query
            if st.button(short, key=f"example_{i}", use_container_width=True):
                st.session_state["search_query"] = query
                st.rerun()
