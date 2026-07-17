"""
Search Page — Fashion Retrieval Streamlit Demo.

Main interactive search interface with query input, results grid,
and per-result score breakdown.
"""
from __future__ import annotations

import os
import sys
import time

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demo.components.result_card import render_result_card
from demo.components.search_box import EXAMPLE_QUERIES, render_search_box
from demo.components.theme import ACCENT_GOLD, PRIMARY_GREEN, apply_custom_theme
from demo.components.utils import show_error, show_no_results

st.set_page_config(
    page_title="Search — Fashion Retrieval",
    page_icon=None,
    layout="wide",
)
apply_custom_theme()


# ── Retriever initialisation (cached) ─────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading models…")
def get_retriever():
    """Load and cache the FashionRetriever (heavy — loads CLIP models)."""
    from part_b_retriever.retriever import FashionRetriever
    return FashionRetriever(device=os.environ.get("DEVICE", "cpu"))


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <h1 style="margin-bottom:0;">Fashion Search</h1>
    <p style="color:#888;margin-top:0;">
        Find fashion images with natural language. Try combining colors, clothing, and context.
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Example query chips ───────────────────────────────────────────────────────
st.markdown(
    f'<p style="color:{ACCENT_GOLD};font-size:0.85rem;font-weight:600;">Quick examples:</p>',
    unsafe_allow_html=True,
)

chip_cols = st.columns(5)
for i, (col, query) in enumerate(zip(chip_cols, EXAMPLE_QUERIES)):
    with col:
        short = query[:28] + "…" if len(query) > 28 else query
        if st.button(short, key=f"chip_{i}", width="stretch"):
            st.session_state["prefill_query"] = query

# ── Search input ──────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill_query", "")

query = st.text_area(
    "Describe what you're looking for:",
    value=prefill,
    placeholder="e.g., A red tie and white shirt in a formal office setting",
    height=90,
    key="main_query",
)

col_k, col_btn = st.columns([3, 1])
with col_k:
    top_k = st.slider("Number of results", 1, 20, 10)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Search", type="primary", width="stretch")

# ── Query components debug panel ──────────────────────────────────────────────
if query and st.toggle("🔬 Show query decomposition", value=False):
    try:
        retriever = get_retriever()
        components = retriever.get_query_components(query)

        with st.expander("Query components", expanded=True):
            col_c, col_cl, col_ctx = st.columns(3)

            with col_c:
                st.markdown(f"**Colors** ({len(components['colors'])})")
                for c in components["colors"]:
                    item_str = f" → {c['item']}" if c.get("item") else ""
                    st.markdown(f"• `{c['color']}`{item_str} ({c['confidence']:.0%})")

            with col_cl:
                st.markdown(f"**Clothing** ({len(components['clothing'])})")
                for cl in components["clothing"]:
                    st.markdown(f"• `{cl['item']}` ({cl['confidence']:.0%})")

            with col_ctx:
                ctx = components["context"]
                st.markdown("**Context**")
                st.markdown(f"• Setting: `{ctx.get('setting', '—')}`")
                st.markdown(f"• Formality: `{ctx.get('formality', 0.5):.0%}`")
                st.markdown(f"• Style: `{components.get('style', '—')}`")

    except Exception as exc:
        st.warning(f"Could not decompose query: {exc}")

# ── Search execution & results ────────────────────────────────────────────────
if search_clicked and query and query.strip():
    st.divider()

    try:
        retriever = get_retriever()
    except Exception as exc:
        show_error(
            f"Failed to load retriever: {exc}. "
            "Make sure PINECONE_API_KEY is set and the index is populated."
        )
        st.stop()

    t0 = time.time()
    with st.spinner("Searching..."):
        try:
            results = retriever.search(query.strip(), top_k=top_k)
        except Exception as exc:
            show_error(str(exc))
            st.stop()

    elapsed_ms = (time.time() - t0) * 1000

    if not results:
        show_no_results()
    else:
        st.markdown(
            f"""
            <div style="margin-bottom:1.5rem;">
                <span style="color:{PRIMARY_GREEN};font-weight:700;font-size:1.2rem;">
                    {len(results)} result{"s" if len(results) != 1 else ""} found
                </span>
                <span style="color:#666;font-size:0.85rem;margin-left:1rem;">
                    in {elapsed_ms:.0f}ms
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Results in a 2-column layout
        left, right = st.columns(2, gap="medium")
        for i, result in enumerate(results):
            with (left if i % 2 == 0 else right):
                render_result_card(rank=i + 1, result=result)

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")
