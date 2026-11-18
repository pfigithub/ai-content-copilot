"""Centralised HTTP client with automatic token refresh.

Every API call MUST route through `request()` so that:

1. A 401 from the backend triggers one refresh attempt and one retry
   of the original request with the new access token.
2. If refresh fails (expired, revoked, reuse detected), the user is
   logged out exactly once — no duplicate "session expired" toasts.
3. All HTTP lives in one place, so changing the refresh strategy later
   requires no changes to the individual components.
"""

from typing import Any

import requests
import streamlit as st

from frontend.config import AUTH_REFRESH, REQUEST_TIMEOUT


def _auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _trigger_logout(reason: str) -> None:
    """Import lazily to avoid a circular import with `frontend.auth`."""
    from frontend.auth import logout

    logout(reason=reason)


def _try_refresh() -> bool:
    """Exchange the current refresh token for a new pair.

    Returns True on success and updates session_state. Returns False on
    any failure (missing token, network error, 4xx). Failures are silent
    — the caller decides whether to force a logout.
    """
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False

    try:
        response = requests.post(
            AUTH_REFRESH,
            json={"refresh_token": refresh_token},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return False

    if not response.ok:
        return False

    data = response.json()
    st.session_state.access_token = data["access_token"]
    st.session_state.refresh_token = data["refresh_token"]
    if data.get("user"):
        st.session_state.user = data["user"]
    return True


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


def _raw_request(
    method: str,
    url: str,
    extra_headers: dict[str, str],
    **kwargs: Any,
) -> requests.Response:
    """Execute a single HTTP request with the current access token."""
    headers = dict(extra_headers)
    headers.update(_auth_headers())
    return requests.request(
        method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs
    )


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Authenticated request with one-shot auto-refresh on 401."""
    # Pop headers once; reuse for the retry so callers keep their
    # Content-Type etc. `kwargs` is consumed by the call itself.
    extra_headers = dict(kwargs.pop("headers", {}))

    try:
        response = _raw_request(method, url, extra_headers, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Connection error: {exc}")

        class _FailedResponse:
            ok = False
            status_code = 0
            text = str(exc)

            @staticmethod
            def json():
                return {"detail": str(exc)}

        return _FailedResponse()  # type: ignore[return-value]

    # Expired access token? Try one refresh, then retry once.
    if response.status_code == 401 and st.session_state.get("refresh_token"):
        if _try_refresh():
            try:
                response = _raw_request(method, url, extra_headers, **kwargs)
            except requests.RequestException as exc:
                st.error(f"Connection error: {exc}")
                return response  # return the original 401 as-is

    # Still 401 → the user must log in again.
    if response.status_code == 401 and st.session_state.get("access_token"):
        _trigger_logout("Your session has expired. Please log in again.")

    return response


def get(url: str, **kwargs: Any) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return request("POST", url, **kwargs)


def delete(url: str, **kwargs: Any) -> requests.Response:
    return request("DELETE", url, **kwargs)