"""Custom CSS injected into every Streamlit page.

Streamlit's defaults are functional but plain. A small amount of CSS
makes the app feel polished without pulling in an external UI library.
"""

import streamlit as st


_CSS = """
<style>
/* ---- Chat ---- */
.stChatMessage {
    padding: 0.85rem 1rem;
    border-radius: 0.65rem;
    margin-bottom: 0.5rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] h3 {
    color: #4a90e2;
    margin-top: 0.5rem;
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 0.5rem;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    border-radius: 0.5rem 0.5rem 0 0;
}

/* ---- Titles ---- */
h1 {
    margin-bottom: 0.25rem;
}

/* ---- Info boxes ---- */
.stAlert {
    border-radius: 0.5rem;
}
</style>
"""


def apply() -> None:
    """Inject the shared stylesheet. Call once per page load."""
    st.markdown(_CSS, unsafe_allow_html=True)