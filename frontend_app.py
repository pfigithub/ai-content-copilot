import streamlit as st
import requests
import uuid

# Configuration
API_URL_CHAT = "http://127.0.0.1:8080/api/v1/chat/"
API_URL_CONTENT = "http://127.0.0.1:8080/api/v1/content/analyze-content"
API_URL_PDF = "http://127.0.0.1:8080/api/v1/content/analyze-pdf"
INGEST_URL = "http://127.0.0.1:8080/api/v1/documents/ingest-pdf"

st.set_page_config(page_title="AI Content Copilot", page_icon="🤖", layout="wide")

st.title("🤖 AI Content Copilot")
st.markdown("Your local, private AI assistant for research, analysis, and document intelligence.")

# Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = f"chat_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("💬 Chat Management")
    current_thread = st.text_input("Thread ID (Change to load a different chat):", value=st.session_state.session_id)
    
    # Reset screen when Thread ID changes
    if current_thread != st.session_state.session_id:
        st.session_state.session_id = current_thread
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Start New Chat"):
        st.session_state.session_id = f"chat_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("🗄️ Permanent Knowledge Base")
    st.info("Files uploaded here are saved to the Qdrant database permanently for the RAG agent.")
    uploaded_file = st.file_uploader("Upload PDF to Database", type="pdf", key="permanent_upload")
    if uploaded_file and st.button("Save to Database"):
        with st.spinner("Ingesting document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(INGEST_URL, files=files)
                if response.status_code == 200:
                    st.success(f"Successfully saved {uploaded_file.name} to database.")
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Main Interface Tabs ---
tab_chat, tab_summarize, tab_analyze, tab_ideas = st.tabs([
    "💬 Smart Chat", "📝 Summarize", "🔍 Analyze", "💡 Ideas"
])

# 1. Chat Tab
with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents, live news, or general knowledge..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    payload = {"query": prompt, "user_id": st.session_state.session_id}
                    response = requests.post(API_URL_CHAT, json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer.")
                        message_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# Helper function for temporary content processing
def process_content(input_type, content_val, task, instruction=""):
    with st.spinner(f"Processing {task}..."):
        try:
            if input_type == "PDF":
                files = {"file": (content_val.name, content_val.getvalue(), "application/pdf")}
                data_payload = {"task": task}
                if instruction:
                    data_payload["instruction"] = instruction
                response = requests.post(API_URL_PDF, files=files, data=data_payload)
            else:
                json_payload = {"task": task}
                if input_type == "Text":
                    json_payload["text"] = content_val
                else:
                    json_payload["url"] = content_val
                if instruction:
                    json_payload["instruction"] = instruction
                response = requests.post(API_URL_CONTENT, json=json_payload)

            if response.status_code == 200:
                data = response.json()
                st.subheader(data.get("title", "Result"))
                if task == "summarize":
                    st.write(data.get("summary"))
                elif task == "analyze":
                    st.write(data.get("analysis"))
                elif task == "ideas":
                    st.write(data.get("summary"))
                    st.write("### Ideas:")
                    for idea in data.get("ideas", []):
                        st.markdown(f"- {idea}")
                
                importance = data.get('importance')
                importance_str = importance if importance is not None else "N/A"
                st.info(f"Category: {data.get('category')} | Importance: {importance_str}")
            else:
                try:
                    error_detail = response.json().get("detail", response.text)
                except:
                    error_detail = response.text
                st.error(f"Error: {error_detail}")
        except Exception as e:
            st.error(f"Connection error: {e}")

def render_content_tab(task_name, description):
    st.markdown(f"#### {description}")
    st.info("Files processed here are NOT saved to the database. They are read once and discarded.")
    input_type = st.radio("Input Type", ["Text", "URL", "PDF"], key=f"{task_name}_type", horizontal=True)
    
    content_val = None
    if input_type == "Text":
        content_val = st.text_area("Enter Text", key=f"{task_name}_val")
    elif input_type == "URL":
        content_val = st.text_input("Enter URL", key=f"{task_name}_url")
    elif input_type == "PDF":
        content_val = st.file_uploader("Upload Temporary PDF", type="pdf", key=f"{task_name}_pdf")
        
    inst = st.text_input("Optional Instructions (e.g., 'Focus on financial data')", key=f"{task_name}_inst")
    
    if st.button(f"Generate {task_name.capitalize()}"):
        if content_val: 
            process_content(input_type, content_val, task_name, inst)
        else:
            st.warning("Please provide input content.")

# 2. Summarize Tab
with tab_summarize:
    render_content_tab("summarize", "Temporary Summarization")

# 3. Analyze Tab
with tab_analyze:
    render_content_tab("analyze", "Temporary Deep Analysis")

# 4. Ideas Tab
with tab_ideas:
    render_content_tab("ideas", "Temporary Brainstorming")