"""
UI utility helpers for the demo.
"""
from __future__ import annotations

import streamlit as st


def show_loading_spinner(message: str = "Searching…"):
    """Context manager that shows a green spinner."""
    return st.spinner(message)


def show_error(message: str) -> None:
    """Display a styled error message."""
    st.error(message)


def show_success(message: str) -> None:
    """Display a styled success message."""
    st.success(message)


def show_no_results() -> None:
    """Display a friendly no-results message."""
    st.markdown(
        """
        <div style="text-align:center;padding:3rem;color:#666;">
            <h3 style="color:#888;">No results found</h3>
            <p>Try a different query or check that the index is populated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
