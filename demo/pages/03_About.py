"""
About Page — Fashion Retrieval Streamlit Demo.

Architecture overview, scoring explanation, and system metrics.
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
    page_title="About — Fashion Retrieval",
    page_icon=None,
    layout="wide",
)
apply_custom_theme()

st.markdown("<h1>About the System</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{TEXT_MUTED}'>Technical overview of the Fashion Attribute Pyramid architecture.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Architecture ──────────────────────────────────────────────────────────────

st.markdown("## Architecture")
st.markdown(
    f"""
    <div class="result-card">
        <pre style="color:#ccc;font-size:0.8rem;background:transparent;border:none;">
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
    │  (color/item/ │
    │   setting/    │
    │   formality)  │
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
    """,
    unsafe_allow_html=True,
)

# ── Scoring explanation ───────────────────────────────────────────────────────

st.markdown("## Scoring Weights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="result-card" style="text-align:center;">
            <div style="font-size:2rem;color:{PRIMARY_GREEN};font-weight:800;">50%</div>
            <h4>Semantic (CLIP)</h4>
            <p style="color:{TEXT_MUTED};font-size:0.85rem;">
                Global scene + context understanding.
                Cosine similarity of CLIP text/image embeddings (ViT-B/32).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="result-card" style="text-align:center;">
            <div style="font-size:2rem;color:{ACCENT_GOLD};font-weight:800;">30%</div>
            <h4>Fashion (FashionCLIP)</h4>
            <p style="color:{TEXT_MUTED};font-size:0.85rem;">
                Fashion-domain fine-tuned encoder.
                Better at clothing item recognition and style nuances.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="result-card" style="text-align:center;">
            <div style="font-size:2rem;color:#E74C3C;font-weight:800;">20%</div>
            <h4>Attributes</h4>
            <p style="color:{TEXT_MUTED};font-size:0.85rem;">
                Explicit attribute matching: color, clothing item, setting, formality.
                Resolves compositionality failures in vanilla CLIP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Key innovations ───────────────────────────────────────────────────────────

st.markdown("## Key Innovations")

innovations = [
    ("Compositional Color-Item Binding", "Associates 'red' with 'tie' and 'white' with 'shirt' — not just 'red' and 'white' in a bag-of-words way."),
    ("Context-Aware Retrieval", "Detects location (office, park, beach) and formality levels from the query text."),
    ("Multi-Vector Search", "Parallel Pinecone queries with CLIP and FashionCLIP, merged with union scoring."),
    ("Hard Constraint Filtering", "Very formal queries enforce a formality_score threshold — casual images are filtered out."),
    ("Diversity Re-ranking", "Greedy MMR-like diversification prevents consecutive same-style results."),
    ("Explainability", "Every result includes a score breakdown and human-readable explanation."),
]

for title, desc in innovations:
    st.markdown(
        f"""
        <div class="result-card" style="margin-bottom:0.75rem;">
            <strong style="color:{PRIMARY_GREEN};">{title}</strong>
            <p style="color:{TEXT_MUTED};font-size:0.85rem;margin:4px 0 0;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Evaluation queries ────────────────────────────────────────────────────────

st.markdown("## Evaluation Queries")

queries = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
]

for i, q in enumerate(queries, 1):
    st.markdown(
        f"""
        <div style="
            display:flex;align-items:center;gap:12px;
            padding:10px 16px;margin-bottom:8px;
            background:#1a1a1a;border-radius:8px;
            border-left:3px solid {ACCENT_GOLD if i == 5 else PRIMARY_GREEN};
        ">
            <span style="color:{ACCENT_GOLD};font-weight:700;min-width:24px;">#{i}</span>
            <span style="font-family:monospace;color:#ccc;font-size:0.9rem;">{q}</span>
            {"<span style='color:#E74C3C;font-size:0.75rem;font-weight:700;'>CORE TEST</span>" if i == 5 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
