"""Admin panel: manage user accounts across their lifecycle."""

import streamlit as st

from frontend.api_client import delete, get, post, show_error
from frontend.config import AUTH_ADMIN_USERS


def _fetch_users(status: str) -> list[dict]:
    response = get(AUTH_ADMIN_USERS, params={"status_filter": status})
    if not response.ok:
        show_error(response)
        return []
    return response.json()


def _render_section(
    status: str,
    action_label: str,
    action_path: str,
    empty_msg: str,
    show_delete: bool = False,
) -> None:
    users = _fetch_users(status)

    if not users:
        st.info(empty_msg)
        return

    for user in users:
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            st.markdown(
                f"**{user['username']}** · `{user['role']}` · "
                f"status: `{user['status']}` · id: `{user['id'][:8]}…`"
            )

        with col2:
            if st.button(
                action_label,
                key=f"{status}_{user['id']}_{action_label}",
                use_container_width=True,
            ):
                resp = post(f"{AUTH_ADMIN_USERS}/{user['id']}/{action_path}")
                if resp.ok:
                    st.success(f"{action_label} succeeded.")
                    st.rerun()
                else:
                    show_error(resp)

        with col3:
            if show_delete and st.button(
                "Delete",
                key=f"{status}_{user['id']}_delete",
                use_container_width=True,
            ):
                resp = delete(f"{AUTH_ADMIN_USERS}/{user['id']}")
                if resp.status_code == 204:
                    st.success("User deleted.")
                    st.rerun()
                else:
                    show_error(resp)


def render() -> None:
    st.markdown("#### 🛡️ User Management")
    st.caption(
        "Approve new accounts, suspend active users, or delete accounts."
    )

    tab_pending, tab_active, tab_suspended = st.tabs(
        ["⏳ Pending", "✅ Active", "⛔ Suspended"]
    )

    with tab_pending:
        _render_section(
            status="pending",
            action_label="Approve",
            action_path="approve",
            empty_msg="No accounts awaiting approval.",
        )

    with tab_active:
        _render_section(
            status="active",
            action_label="Suspend",
            action_path="suspend",
            empty_msg="No active users.",
        )

    with tab_suspended:
        _render_section(
            status="suspended",
            action_label="Reactivate",
            action_path="approve",
            empty_msg="No suspended users.",
            show_delete=True,
        )