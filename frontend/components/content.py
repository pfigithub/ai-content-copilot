"""Summarize / Analyze / Ideas tabs — all share one input form."""

import streamlit as st

from frontend.api_client import post, show_error
from frontend.config import CONTENT_ANALYZE, CONTENT_PDF


_LABELS = {
    "summarize": "Summarize",
    "analyze": "Analyze",
    "ideas": "Ideas",
}

_DESCRIPTIONS = {
    "summarize": "Get a faithful summary of the source.",
    "analyze": "Deep analysis with key insights and importance.",
    "ideas": "Brainstorm related concepts and creative directions.",
}


def render(task: str) -> None:
    st.markdown(f"#### 📌 {_LABELS[task]}")
    st.caption(_DESCRIPTIONS[task])
    st.info(
        "Content processed here is temporary and is **not** saved to "
        "your knowledge base."
    )

    input_type = st.radio(
        "Input Type",
        ["Text", "URL", "PDF"],
        key=f"{task}_type",
        horizontal=True,
    )

    content_value = None
    if input_type == "Text":
        content_value = st.text_area("Paste your text", key=f"{task}_text")
    elif input_type == "URL":
        content_value = st.text_input("URL", key=f"{task}_url")
    else:
        content_value = st.file_uploader(
            "Upload a PDF", type=["pdf"], key=f"{task}_pdf"
        )

    instruction = st.text_input(
        "Optional instruction (language, tone, focus...)",
        key=f"{task}_instruction",
    )

    if not st.button(
        f"Generate {_LABELS[task]}", key=f"{task}_submit"
    ):
        return

    if content_value is None or (
        isinstance(content_value, str) and not content_value.strip()
    ):
        st.warning("Please provide valid input.")
        return

    with st.spinner(f"Running {task}..."):
        if input_type == "PDF":
            files = {
                "file": (
                    content_value.name,
                    content_value.getvalue(),
                    "application/pdf",
                )
            }
            data = {"task": task}
            if instruction.strip():
                data["instruction"] = instruction.strip()
            response = post(CONTENT_PDF, files=files, data=data)
        else:
            payload = {"task": task}
            if input_type == "Text":
                payload["text"] = content_value
            else:
                payload["url"] = content_value
            if instruction.strip():
                payload["instruction"] = instruction.strip()
            response = post(CONTENT_ANALYZE, json=payload)

    if not response.ok:
        show_error(response)
        return

    result = response.json()
    st.subheader(result.get("title", "Result"))

    if task == "summarize":
        st.write(result.get("summary", ""))
    elif task == "analyze":
        st.write(result.get("analysis", ""))
    elif task == "ideas":
        st.write(result.get("summary", ""))
        st.markdown("### 💡 Ideas")
        for idea in result.get("ideas", []):
            st.markdown(f"- {idea}")

    importance = result.get("importance")
    st.info(
        f"**Category:** {result.get('category', 'other')} · "
        f"**Importance:** {importance if importance is not None else 'n/a'}"
    )