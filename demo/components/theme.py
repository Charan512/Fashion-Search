"""
Theme configuration for the Fashion Retrieval Streamlit demo.

Applies the green / gold / black color scheme via injected CSS.
"""
from __future__ import annotations

import streamlit as st

# ── Color constants ───────────────────────────────────────────────────────────

PRIMARY_GREEN = "#2ECC71"
ACCENT_GOLD = "#F39C12"
DARK_BG = "#0d0d0d"
DARK_SECONDARY = "#1a1a1a"
DARK_CARD = "#222222"
TEXT_WHITE = "#F0F0F0"
TEXT_MUTED = "#A0A0A0"
SUCCESS_GREEN = "#27AE60"
ERROR_RED = "#C0392B"
BORDER_GOLD = "#B8860B"


def apply_custom_theme() -> None:
    """Inject green / gold / black CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Root variables ── */
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

        /* ── Global ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: var(--text-white);
        }}

        .stApp {{
            background: linear-gradient(135deg, {DARK_BG} 0%, #111111 100%);
            min-height: 100vh;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background-color: {DARK_SECONDARY} !important;
            border-right: 2px solid {ACCENT_GOLD};
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--text-white) !important;
        }}

        /* ── Headings ── */
        h1, h2, h3 {{
            color: var(--primary-green) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}

        h4, h5, h6 {{
            color: var(--accent-gold) !important;
            font-weight: 600 !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg, {PRIMARY_GREEN}, {SUCCESS_GREEN}) !important;
            color: #000 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 0.6rem 1.4rem !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3) !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(46, 204, 113, 0.5) !important;
        }}

        /* ── Text inputs / text areas ── */
        .stTextArea textarea, .stTextInput input {{
            background-color: {DARK_CARD} !important;
            color: var(--text-white) !important;
            border: 1px solid {BORDER_GOLD} !important;
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
        }}

        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: var(--primary-green) !important;
            box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.2) !important;
        }}

        /* ── Sliders ── */
        .stSlider [data-baseweb="slider"] [data-testid="stSliderThumb"] {{
            background-color: var(--accent-gold) !important;
        }}

        /* ── Metric widgets ── */
        [data-testid="stMetricValue"] {{
            color: var(--primary-green) !important;
            font-weight: 700 !important;
        }}

        /* ── Dividers ── */
        hr {{
            border-color: {BORDER_GOLD} !important;
            opacity: 0.4;
        }}

        /* ── Expanders ── */
        details summary {{
            color: var(--accent-gold) !important;
            font-weight: 600 !important;
        }}

        /* ── Success / info / warning banners ── */
        .stSuccess {{
            background-color: rgba(46, 204, 113, 0.1) !important;
            border-left: 4px solid var(--primary-green) !important;
        }}

        .stInfo {{
            background-color: rgba(243, 156, 18, 0.1) !important;
            border-left: 4px solid var(--accent-gold) !important;
        }}

        /* ── Result cards ── */
        .result-card {{
            background: var(--dark-card);
            border: 1px solid var(--border-gold);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }}

        .result-card:hover {{
            border-color: var(--primary-green);
            box-shadow: 0 4px 20px rgba(46, 204, 113, 0.15);
            transform: translateY(-2px);
        }}

        /* ── Score bars ── */
        .score-bar-container {{
            background: #333;
            border-radius: 4px;
            height: 6px;
            margin-top: 4px;
        }}

        .score-bar-fill {{
            height: 6px;
            border-radius: 4px;
            background: linear-gradient(90deg, {PRIMARY_GREEN}, {ACCENT_GOLD});
            transition: width 0.4s ease;
        }}

        /* ── Attribute badges ── */
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

        /* ── Hero search area ── */
        .search-hero {{
            background: linear-gradient(135deg,
                rgba(46, 204, 113, 0.05) 0%,
                rgba(243, 156, 18, 0.05) 100%
            );
            border: 1px solid rgba(243, 156, 18, 0.2);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
