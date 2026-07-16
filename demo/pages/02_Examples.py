"""
Examples Page — Fashion Retrieval Streamlit Demo.

Shows all 5 evaluation queries with descriptions of what they test.
Users can click to pre-fill the search page.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demo.components.theme import ACCENT_GOLD, PRIMARY_GREEN, TEXT_MUTED, apply_custom_theme

st.set_page_config(
    page_title="Examples — Fashion Retrieval",
    page_icon=None,
    layout="wide",
)
apply_custom_theme()

EVALUATION_QUERIES = [
    {
        "query": "A person in a bright yellow raincoat",
        "title": "Query 1 — Attribute Specific",
        "description": "Tests color classification and single-item focus. The system must return images with bright yellow outerwear, not orange or green.",
        "tests": ["Color precision", "Single attribute", "Clothing type"],
    },
    {
        "query": "Professional business attire inside a modern office",
        "title": "Query 2 — Contextual / Place",
        "description": "Tests context understanding and formality classification. Results must show formal office wear in a corporate setting.",
        "tests": ["Setting detection", "Formality scoring", "Context match"],
    },
    {
        "query": "Someone wearing a blue shirt sitting on a park bench",
        "title": "Query 3 — Complex Semantic",
        "description": "Tests multi-attribute + pose + location awareness. Combines color, clothing, setting, and position.",
        "tests": ["Multi-attribute", "Location (park)", "Color + item"],
    },
    {
        "query": "Casual weekend outfit for a city walk",
        "title": "Query 4 — Style Inference",
        "description": "Tests style classification without explicit item names. The system must infer 'casual' from semantic context.",
        "tests": ["Style inference", "Implicit attributes", "Urban context"],
    },
    {
        "query": "A red tie and a white shirt in a formal setting",
        "title": "Query 5 — Compositional (CORE)",
        "description": "Core compositionality test. Must distinguish 'red tie + white shirt' from 'white tie + red shirt'. This is where vanilla CLIP fails.",
        "tests": ["Color-item composition", "Formality", "Multi-garment"],
    },
]

st.markdown("<h1>Evaluation Examples</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{TEXT_MUTED}'>These are the 5 official evaluation queries used to assess the system. "
    "Each tests a different capability of the Fashion Attribute Pyramid.</p>",
    unsafe_allow_html=True,
)
st.divider()

for i, eq in enumerate(EVALUATION_QUERIES):
    with st.container():
        st.markdown(
            f"""
            <div class="result-card" style="margin-bottom:1.5rem;">
                <div style="margin-bottom:12px;">
                    <h4 style="margin:0;color:{ACCENT_GOLD};">{eq['title']}</h4>
                    <p style="margin:4px 0 0;color:{TEXT_MUTED};font-size:0.85rem;">{eq['description']}</p>
                </div>
                <div style="
                    background:#111;
                    border-left:3px solid {PRIMARY_GREEN};
                    padding:10px 14px;
                    border-radius:4px;
                    font-family:monospace;
                    color:{PRIMARY_GREEN};
                    margin-bottom:12px;
                    font-size:0.9rem;
                ">
                    "{eq['query']}"
                </div>
                <div>
            """,
            unsafe_allow_html=True,
        )
        for test in eq["tests"]:
            st.markdown(
                f'<span class="attr-badge">✓ {test}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

        if st.button(f"Run Query {i+1}", key=f"run_{i}", use_container_width=False):
            st.session_state["prefill_query"] = eq["query"]
            st.switch_page("pages/01_Search.py")

        st.markdown("<br>", unsafe_allow_html=True)
