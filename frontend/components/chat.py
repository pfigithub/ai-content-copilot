"""Chat tab: conversation history and input."""

import streamlit as st

from frontend.api_client import post, show_error
from frontend.config import CHAT_URL


def render() -> None:
    st.markdown("#### 💬 Smart Chat")
    st.caption(
        "Ask about your indexed documents, live web data, or general "
        "knowledge. The assistant decides when to use tools."
    )

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask anything...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = post(
                CHAT_URL,
                json={
                    "query": prompt,
                    # Server combines this with the authenticated user id.
                    "user_id": st.session_state.get("session_id")
                    or "default",
                },
            )

        if response.ok:
            answer = response.json().get("answer", "No answer.")
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            show_error(response)