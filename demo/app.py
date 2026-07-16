"""
Fashion Retrieval — Streamlit App Entry Point.

Run with:
    streamlit run demo/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure project root is on Python path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demo.components.theme import apply_custom_theme

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Fashion Retrieval System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Fashion Context Retrieval — Multi-Vector Search with CLIP + FashionCLIP",
    },
)

apply_custom_theme()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0;">
            <h2 style="margin:0;">Fashion<br>Retrieval</h2>
            <p style="color:#888;font-size:0.8rem;margin-top:0.5rem;">
                Multi-Vector Fashion Search
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Navigation")
    st.page_link("pages/01_Search.py", label="Search")
    st.page_link("pages/02_Examples.py", label="Examples")
    st.page_link("pages/03_About.py", label="About")

    st.divider()
    st.markdown(
        "<p style='color:#555;font-size:0.75rem;'>Fashion Attribute Pyramid · CLIP + FashionCLIP · Pinecone</p>",
        unsafe_allow_html=True,
    )

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:4rem 2rem;">
        <h1 style="font-size:3rem;margin-bottom:0.5rem;">Fashion Retrieval System</h1>
        <p style="color:#888;font-size:1.1rem;max-width:600px;margin:0 auto 2rem;">
            Find fashion images using natural language descriptions.
            Powered by multi-vector search with CLIP, FashionCLIP,
            and explicit attribute decomposition.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="result-card" style="text-align:center;">
            <h4>Natural Language Search</h4>
            <p style="color:#888;font-size:0.85rem;">
                Describe outfits, colors, settings, and styles in plain English.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="result-card" style="text-align:center;">
            <h4>Multi-Vector Retrieval</h4>
            <p style="color:#888;font-size:0.85rem;">
                CLIP (50%) + FashionCLIP (30%) + Attribute matching (20%)
                beats vanilla CLIP on compositionality.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="result-card" style="text-align:center;">
            <h4>Explainable Results</h4>
            <p style="color:#888;font-size:0.85rem;">
                Every result shows exactly why it matched:
                score breakdown + matched attributes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Start Searching", type="primary"):
    st.switch_page("pages/01_Search.py")
