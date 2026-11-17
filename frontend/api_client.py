"""Centralised HTTP client.

Every API call MUST route through `request()` so a single place
handles 401 (expired token) by logging the user out. Otherwise every
component would need its own 401 handling, and one forgotten case
would break the app in a confusing way.
"""

from typing import Any

import requests
import streamlit as st

from frontend.config import REQUEST_TIMEOUT


def _auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _trigger_logout(reason: str) -> None:
    """Import lazily to avoid a circular dependency with `frontend.auth`."""
    from frontend.auth import logout

    logout(reason=reason)


def show_error(response: requests.Response) -> None:
    """Display a user-friendly error without exposing credentials."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    if response.status_code == 400:
        st.error(f"Invalid request: {detail}")
    elif response.status_code == 403:
        st.error(f"Access denied: {detail}")
    elif response.status_code == 404:
        st.error("The requested resource was not found.")
    elif response.status_code == 422:
        st.error(f"Validation error: {detail}")
    elif response.status_code == 502:
        st.error(f"Upstream service error: {detail}")
    elif response.status_code == 503:
        st.error("The service is temporarily unavailable.")
    else:
        st.error(f"API error ({response.status_code}): {detail}")


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Send an authenticated request; auto-logout on token expiry.

    On network failure, returns a synthetic 0-status response so callers
    can uniformly check `.ok` without wrapping every call in try/except.
    """
    headers = dict(kwargs.pop("headers", {}))
    headers.update(_auth_headers())

    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as exc:
        st.error(f"Connection error: {exc}")
        # Return a sentinel that has .ok == False and .status_code == 0.
        class _FailedResponse:
            ok = False
            status_code = 0
            text = str(exc)

            @staticmethod
            def json():
                return {"detail": str(exc)}
        return _FailedResponse()  # type: ignore[return-value]

    if response.status_code == 401 and st.session_state.get("access_token"):
        _trigger_logout("Your session has expired. Please log in again.")

    return response


def get(url: str, **kwargs: Any) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return request("POST", url, **kwargs)


def delete(url: str, **kwargs: Any) -> requests.Response:
    return request("DELETE", url, **kwargs)