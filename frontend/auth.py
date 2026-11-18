"""Authentication flow: login, register, session guard, inactivity.

Token storage: both access and refresh tokens live in
`st.session_state`, which is server-side. Nothing is written to a
cookie or to localStorage, so a browser inspector cannot read them.
"""

import time

import requests
import streamlit as st

from frontend.config import (
    AUTH_LOGIN,
    AUTH_ME,
    AUTH_REGISTER,
    INACTIVITY_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT,
)


_DEFAULTS = {
    "access_token": None,
    "refresh_token": None,
    "user": None,
    "session_id": None,
    "messages": [],
    "last_activity": None,
}


def _init_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                list(value) if isinstance(value, list) else value
            )


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def is_admin() -> bool:
    user = st.session_state.get("user") or {}
    return user.get("role") == "admin"


def logout(reason: str | None = None) -> None:
    """Revoke the refresh token server-side, clear local state, rerun."""
    refresh_token = st.session_state.get("refresh_token")
    if refresh_token:
        try:
            from frontend.config import AUTH_LOGOUT

            requests.post(
                AUTH_LOGOUT,
                json={"refresh_token": refresh_token},
                # Short timeout: logout should never block the UI.
                timeout=(2, 5),
            )
        except requests.RequestException:
            # Revocation failure is non-fatal. The token will expire
            # eventually; we still clear local state so the UI resets.
            pass

    for key in list(_DEFAULTS.keys()):
        st.session_state.pop(key, None)
    if reason:
        st.session_state["_logout_reason"] = reason
    st.rerun()


def _touch() -> None:
    st.session_state.last_activity = time.time()


def _check_inactivity() -> None:
    last = st.session_state.get("last_activity")
    if last is None:
        _touch()
        return
    if time.time() - last > INACTIVITY_TIMEOUT_SECONDS:
        logout(reason="You were logged out after a period of inactivity.")


def _verify_token() -> None:
    """Confirm the session is valid via /auth/me.

    Uses the centralised api_client so an expired access token is
    transparently refreshed. The api_client calls `logout` itself if
    refresh also fails, so the only 4xx we still handle here is 403
    (suspended / deleted account).
    """
    from frontend.api_client import get as api_get

    response = api_get(AUTH_ME)

    if response.ok:
        st.session_state.user = response.json()
        return

    if response.status_code == 403:
        logout(reason="Your account is no longer active.")


def _render_login() -> None:
    st.subheader("🔐 Login")

    with st.form("login_form"):
        username = st.text_input("Username", max_chars=64)
        password = st.text_input("Password", type="password", max_chars=128)
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if not submitted:
        return

    if not username.strip() or not password:
        st.error("Username and password are required.")
        return

    try:
        response = requests.post(
            AUTH_LOGIN,
            json={"username": username.strip(), "password": password},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        st.error(f"Connection error: {exc}")
        return

    if response.status_code == 200:
        data = response.json()
        st.session_state.access_token = data["access_token"]
        st.session_state.refresh_token = data.get("refresh_token")
        st.session_state.user = data.get("user")
        st.session_state.last_activity = time.time()
        st.session_state.pop("session_id", None)
        st.session_state.pop("messages", None)
        st.rerun()

    if response.status_code == 401:
        st.error("Invalid username or password.")
        return

    if response.status_code == 403:
        detail = response.json().get("detail", "").lower()
        if "pending" in detail:
            st.warning(
                "Your account is pending admin approval. "
                "Please check back later."
            )
        elif "suspend" in detail:
            st.error("Your account has been suspended.")
        else:
            st.error("Access denied.")
        return

    if response.status_code == 503:
        st.error("Authentication is not configured on the server.")
        return

    st.error(f"Unexpected error ({response.status_code}).")


def _render_register() -> None:
    st.subheader("📝 Create Account")
    st.info(
        "After registering, an administrator must approve your account "
        "before you can log in."
    )

    with st.form("register_form"):
        username = st.text_input(
            "Username",
            max_chars=64,
            help="3-64 characters: letters, numbers, underscores.",
        )
        password = st.text_input(
            "Password",
            type="password",
            max_chars=128,
            help="At least 8 characters.",
        )
        confirm = st.text_input(
            "Confirm Password",
            type="password",
            max_chars=128,
        )
        submitted = st.form_submit_button(
            "Create account", use_container_width=True
        )

    if not submitted:
        return

    if not username.strip():
        st.error("Username is required.")
        return

    if len(password) < 8:
        st.error("Password must be at least 8 characters.")
        return

    if password != confirm:
        st.error("Passwords do not match.")
        return

    try:
        response = requests.post(
            AUTH_REGISTER,
            json={"username": username.strip(), "password": password},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        st.error(f"Connection error: {exc}")
        return

    if response.status_code == 201:
        st.success(
            "Account created. An administrator must approve it before "
            "you can log in."
        )
        return

    if response.status_code == 409:
        st.error("That username is already taken.")
        return

    if response.status_code == 422:
        st.error("Invalid input. Check the username and password rules.")
        return

    st.error(f"Unexpected error ({response.status_code}).")


def render_gate() -> None:
    """Enforce auth and inactivity, else render login. Calls st.stop()."""
    _init_state()

    reason = st.session_state.pop("_logout_reason", None)
    if reason:
        st.info(reason)

    if is_authenticated():
        _check_inactivity()
        if is_authenticated():
            _verify_token()
        if is_authenticated():
            _touch()

    if is_authenticated():
        return

    st.title("🤖 AI Content Copilot")
    st.caption("A private, local-first AI research assistant.")

    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Create Account"])
    with tab_login:
        _render_login()
    with tab_register:
        _render_register()

    st.stop()