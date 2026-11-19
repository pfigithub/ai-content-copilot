"""Profile tab: view account info and change password."""

import streamlit as st

from frontend.api_client import post, show_error
from frontend.auth import logout
from frontend.config import AUTH_CHANGE_PASSWORD


def _render_account_info() -> None:
    user = st.session_state.get("user") or {}
    st.markdown("#### 👤 Account")
    st.markdown(f"**Username:** `{user.get('username', 'unknown')}`")
    st.markdown(f"**Role:** `{user.get('role', 'unknown')}`")
    st.markdown(f"**Status:** `{user.get('status', 'unknown')}`")
    st.markdown(f"**User ID:** `{user.get('id', 'unknown')}`")


def _render_change_password_form() -> None:
    st.markdown("#### 🔑 Change Password")
    st.caption(
        "After changing your password, you will be logged out of all "
        "devices for security reasons."
    )

    with st.form("change_password_form"):
        current = st.text_input(
            "Current password", type="password", max_chars=128
        )
        new = st.text_input(
            "New password",
            type="password",
            max_chars=128,
            help="At least 8 characters.",
        )
        confirm = st.text_input(
            "Confirm new password", type="password", max_chars=128
        )
        submitted = st.form_submit_button(
            "Change password", use_container_width=True
        )

    if not submitted:
        return

    if not current or not new or not confirm:
        st.error("All fields are required.")
        return

    if len(new) < 8:
        st.error("New password must be at least 8 characters.")
        return

    if new != confirm:
        st.error("New password and confirmation do not match.")
        return

    response = post(
        AUTH_CHANGE_PASSWORD,
        json={"current_password": current, "new_password": new},
    )

    if response.status_code == 204:
        # The backend already revoked all refresh tokens. Clear local
        # state and send the user back to the login page with a message.
        logout(
            reason="Password changed. Please log in with your new password."
        )
        return

    if response.status_code == 400:
        # The backend returns 400 (not 401) for a wrong current password
        # so that api_client's auto-logout on 401 is not triggered.
        try:
            detail = response.json().get("detail", "Invalid request.")
        except ValueError:
            detail = "Invalid request."
        st.error(detail)
        return

    if response.status_code == 422:
        detail = response.json().get("detail", "Invalid input.")
        st.error(detail)
        return

    show_error(response)


def render() -> None:
    _render_account_info()
    st.divider()
    _render_change_password_form()