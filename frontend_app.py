
import os
import uuid

import requests
import streamlit as st


API_BASE = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_URL_CHAT = f"{API_BASE}/api/v1/chat/"
API_URL_CONTENT = f"{API_BASE}/api/v1/content/analyze-content"
API_URL_PDF = f"{API_BASE}/api/v1/content/analyze-pdf"
INGEST_URL = f"{API_BASE}/api/v1/documents/ingest-pdf"

REQUEST_TIMEOUT = (5, 120)


st.set_page_config(
    page_title="AI Content Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Content Copilot")

st.markdown(
    "Your local, private AI assistant for research, analysis, "
    "and document intelligence."
)


# The conversation ID is not an authenticated user identity.
if "session_id" not in st.session_state:
    st.session_state.session_id = f"chat_{uuid.uuid4().hex[:12]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


def auth_headers() -> dict[str, str]:
    """Build the Bearer header from the key entered in this Streamlit session."""
    token = st.session_state.get("api_key", "").strip()

    return {
        "Authorization": f"Bearer {token}"
    } if token else {}


def show_api_error(response: requests.Response) -> None:
    """Display an API error without exposing authentication credentials."""

    if response.status_code == 401:
        st.error(
            "Authentication failed. Check the API key in the sidebar."
        )
        return

    if response.status_code == 503:
        st.error(
            "API authentication or a required backend service is not configured."
        )
        return

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    st.error(
        f"API Error ({response.status_code}): {detail}"
    )


def post_api(url: str, **kwargs) -> requests.Response:
    """Send an authenticated request with a bounded network timeout."""

    headers = dict(kwargs.pop("headers", {}))
    headers.update(auth_headers())

    return requests.post(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )


with st.sidebar:

    st.header("🔐 API Authentication")

    st.text_input(
        "Bearer API Key",
        type="password",
        key="api_key",
        help="Enter the API key generated for your account.",
    )

    st.caption(
        "The key remains in this Streamlit session; "
        "it is not written to project files."
    )

    authenticated = bool(
        st.session_state.get("api_key", "").strip()
    )

    st.divider()

    st.header("💬 Chat Management")

    current_thread = st.text_input(
        "Conversation ID",
        value=st.session_state.session_id,
        disabled=not authenticated,
    )

    if authenticated and current_thread != st.session_state.session_id:
        st.session_state.session_id = current_thread
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🔄 Start New Chat",
        disabled=not authenticated,
    ):
        st.session_state.session_id = (
            f"chat_{uuid.uuid4().hex[:12]}"
        )
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("🗄️ Permanent Knowledge Base")

    st.info(
        "Files uploaded here are stored in your private Qdrant namespace."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF to Database",
        type=["pdf"],
        key="permanent_upload",
        disabled=not authenticated,
    )

    if st.button(
        "Save to Database",
        disabled=not authenticated or uploaded_file is None,
    ):

        with st.spinner("Ingesting document..."):

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }

                response = post_api(
                    INGEST_URL,
                    files=files,
                )

                if response.ok:
                    st.success(
                        f"Successfully saved {uploaded_file.name}."
                    )
                else:
                    show_api_error(response)

            except requests.RequestException as exc:
                st.error(f"Connection error: {exc}")


if not authenticated:
    st.info(
        "Enter your API key in the sidebar to use the copilot."
    )
    st.stop()


tab_chat, tab_summarize, tab_analyze, tab_ideas = st.tabs(
    [
        "💬 Smart Chat",
        "📝 Summarize",
        "🔍 Analyze",
        "💡 Ideas",
    ]
)


# 1. Chat Tab
with tab_chat:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "Ask about your documents, live news, or general knowledge..."
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            placeholder = st.empty()

            with st.spinner("Thinking..."):

                try:
                    payload = {
                        "query": prompt,
                        # The API combines this conversation ID with the authenticated owner.
                        "user_id": st.session_state.session_id,
                    }

                    response = post_api(
                        API_URL_CHAT,
                        json=payload,
                    )

                    if response.ok:

                        answer = response.json().get(
                            "answer",
                            "No answer.",
                        )

                        placeholder.markdown(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                    else:
                        show_api_error(response)

                except requests.RequestException as exc:
                    st.error(f"Connection error: {exc}")


def process_content(
    input_type: str,
    content_value,
    task: str,
    instruction: str = "",
) -> None:
    """Submit temporary content and render its task-specific structured response."""

    with st.spinner(f"Processing {task}..."):

        try:
            clean_instruction = instruction.strip()

            if input_type == "PDF":

                files = {
                    "file": (
                        content_value.name,
                        content_value.getvalue(),
                        "application/pdf",
                    )
                }

                data = {
                    "task": task,
                }

                if clean_instruction:
                    data["instruction"] = clean_instruction

                response = post_api(
                    API_URL_PDF,
                    files=files,
                    data=data,
                )

            else:

                payload = {
                    "task": task,
                }

                if input_type == "Text":
                    payload["text"] = content_value
                else:
                    payload["url"] = content_value

                if clean_instruction:
                    payload["instruction"] = clean_instruction

                response = post_api(
                    API_URL_CONTENT,
                    json=payload,
                )

            if not response.ok:
                show_api_error(response)
                return

            data = response.json()

            st.subheader(
                data.get("title", "Result")
            )

            if task == "summarize":

                st.write(
                    data.get("summary", "")
                )

            elif task == "analyze":

                st.write(
                    data.get("analysis", "")
                )

            elif task == "ideas":

                st.write(
                    data.get("summary", "")
                )

                st.markdown("### Ideas")

                for idea in data.get("ideas", []):
                    st.markdown(f"- {idea}")

            importance = data.get("importance")

            importance_text = (
                str(importance)
                if importance is not None
                else "Not applicable"
            )

            st.info(
                f"Category: {data.get('category', 'other')} | "
                f"Importance: {importance_text}"
            )

        except requests.RequestException as exc:
            st.error(f"Connection error: {exc}")


def render_content_tab(
    task_name: str,
    description: str,
) -> None:
    """Render a temporary text, URL, or PDF input form."""

    st.markdown(f"#### {description}")

    st.info(
        "Content processed here is not saved to the permanent knowledge base."
    )

    input_type = st.radio(
        "Input Type",
        ["Text", "URL", "PDF"],
        key=f"{task_name}_type",
        horizontal=True,
    )

    content_value = None

    if input_type == "Text":

        content_value = st.text_area(
            "Enter Text",
            key=f"{task_name}_value",
        )

    elif input_type == "URL":

        content_value = st.text_input(
            "Enter URL",
            key=f"{task_name}_url",
        )

    else:

        content_value = st.file_uploader(
            "Upload Temporary PDF",
            type=["pdf"],
            key=f"{task_name}_pdf",
        )

    instruction = st.text_input(
        "Optional Instructions",
        key=f"{task_name}_instruction",
    )

    if st.button(
        f"Generate {task_name.capitalize()}"
    ):

        valid_input = (
            content_value is not None
            and (
                not isinstance(content_value, str)
                or bool(content_value.strip())
            )
        )

        if not valid_input:
            st.warning("Please provide valid input content.")
            return

        process_content(
            input_type=input_type,
            content_value=content_value,
            task=task_name,
            instruction=instruction,
        )


with tab_summarize:
    render_content_tab(
        "summarize",
        "Temporary Summarization",
    )


with tab_analyze:
    render_content_tab(
        "analyze",
        "Temporary Deep Analysis",
    )


with tab_ideas:
    render_content_tab(
        "ideas",
        "Temporary Brainstorming",
    )
