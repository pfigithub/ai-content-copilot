"""Streamlit entry point.

This file ONLY wires things together; all logic lives in sibling modules.
Keeping the entry point thin makes it easy to reason about what runs
when and in what order.
"""

import sys
from pathlib import Path

# Make `import frontend.*` work regardless of how Streamlit is invoked.
# Streamlit runs this file with CWD = /app inside Docker, but the import
# behaviour is not guaranteed across versions, so we add the project
# root explicitly. This is a deliberate, minimal bootstrap.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import uuid

import streamlit as st

from frontend import styles
from frontend.auth import is_admin, logout, render_gate
from frontend.components import admin, chat, content, profile


st.set_page_config(
    page_title="AI Content Copilot",
    page_icon="🤖",
    layout="wide",
)

styles.apply()
render_gate()  # blocks execution until the user is authenticated


# ------------------------------------------------------------------ #
# Sidebar: identity, logout, chat session                            #
# ------------------------------------------------------------------ #

user = st.session_state.get("user") or {}
role_label = "🛡️ Admin" if is_admin() else "👤 User"

with st.sidebar:
    st.markdown(f"### 👋 Welcome, **{user.get('username', 'user')}**")
    st.caption(f"{role_label} · status: `{user.get('status', 'unknown')}`")

    if st.button("🚪 Log out", use_container_width=True):
        logout()

    st.divider()

    st.markdown("### 💬 Chat Session")
    if st.session_state.get("session_id") is None:
        st.session_state.session_id = f"chat_{uuid.uuid4().hex[:12]}"

    st.text_input(
        "Conversation ID",
        value=st.session_state.session_id,
        disabled=True,
    )

    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.session_id = f"chat_{uuid.uuid4().hex[:12]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Sessions expire after 30 minutes of inactivity "
        "for security."
    )


# ------------------------------------------------------------------ #
# Main tabs                                                          #
# ------------------------------------------------------------------ #

tab_titles = ["💬 Chat", "📝 Summarize", "🔍 Analyze", "💡 Ideas", " 👤 Profile",]
if is_admin():
    tab_titles.append("🛡️ Admin")

tabs = st.tabs(tab_titles)

with tabs[0]:
    chat.render()
with tabs[1]:
    content.render("summarize")
with tabs[2]:
    content.render("analyze")
with tabs[3]:
    content.render("ideas")
with tabs[4]:
    profile.render()
if is_admin():
    with tabs[5]:
        admin.render()