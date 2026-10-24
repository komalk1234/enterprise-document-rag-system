"""
app.py - Enterprise Document Intelligence – Streamlit Frontend

Handles:
- Streamlit configuration and styling
- Sidebar with document + system info
- PDF upload and processing
- Chat interface (question input + history)
- Displaying answers and source citations
- Error messages for common failure modes
"""

import warnings
# Suppress cosmetic deprecation notices from transitional langchain packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*")

import streamlit as st
import time
from pathlib import Path

from rag_pipeline import (
    process_pdf,
    generate_answer,
    check_ollama_connection,
    OLLAMA_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from utils import (
    validate_pdf,
    save_uploaded_file,
    cleanup_temp_file,
    format_page_numbers,
    truncate_text,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS – clean, modern look without fragile hacks
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Main title ── */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6c63ff, #48cfad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8892b0;
        font-size: 1.05rem;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    /* ── Status / info cards ── */
    .doc-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .status-success {
        background: rgba(72,207,173,0.12);
        border: 1px solid rgba(72,207,173,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #48cfad;
        font-weight: 500;
    }

    .status-error {
        background: rgba(255,82,82,0.12);
        border: 1px solid rgba(255,82,82,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #ff5252;
        font-weight: 500;
    }

    /* ── Chat message bubbles ── */
    .user-message {
        background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(108,99,255,0.1));
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 12px 12px 4px 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        color: #e2e2f0;
    }

    .assistant-message {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(72,207,173,0.25);
        border-radius: 12px 12px 12px 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        color: #ccd6f6;
    }

    .source-badge {
        display: inline-block;
        background: rgba(108,99,255,0.2);
        border: 1px solid rgba(108,99,255,0.5);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #6c63ff;
        margin: 2px 3px;
        font-weight: 500;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(108,99,255,0.2);
    }

    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #6c63ff;
        margin-bottom: 0.5rem;
    }

    .sidebar-label {
        font-size: 0.78rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
    }

    .sidebar-value {
        color: #ccd6f6;
        font-size: 0.9rem;
    }

    /* ── Divider ── */
    .fancy-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108,99,255,0.5), transparent);
        margin: 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "chat_history":  [],       # list of {"role": "user"|"assistant", "content": ..., "sources": ...}
        "vector_store":  None,     # Chroma instance after processing
        "doc_info":      None,     # {"filename", "pages", "chunks"}
        "processing":    False,    # True whilst PDF is being processed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-header">🏢 Enterprise RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Document Information ──
    st.markdown('<div class="sidebar-label">📄 Document Information</div>', unsafe_allow_html=True)

    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown(f'<div class="sidebar-value">📎 <b>File:</b> {info["filename"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-value">📃 <b>Pages:</b> {info["pages"]}</div>',    unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-value">🧩 <b>Chunks:</b> {info["chunks"]}</div>',  unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-value" style="color:#556;">No document loaded</div>', unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── System Information ──
    st.markdown('<div class="sidebar-label">⚙️ System Information</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">🤖 <b>LLM:</b> Llama 3.2 3B</div>',              unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">🔢 <b>Embeddings:</b> all-MiniLM-L6-v2</div>',   unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">🗄️ <b>Vector DB:</b> ChromaDB</div>',             unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">🔗 <b>Framework:</b> LangChain</div>',            unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">🖥️ <b>Chunk Size:</b> {CHUNK_SIZE} chars</div>',  unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">🔄 <b>Overlap:</b> {CHUNK_OVERLAP} chars</div>', unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Ollama Status ──
    st.markdown('<div class="sidebar-label">🔌 Ollama Status</div>', unsafe_allow_html=True)
    if st.button("Check Ollama", use_container_width=True, key="check_ollama_btn"):
        with st.spinner("Checking…"):
            ok, msg = check_ollama_connection()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Clear Chat ──
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_sidebar"):
        st.session_state.chat_history = []
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA – Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">🏢 Enterprise Document Intelligence</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions and get AI-powered answers grounded in your documents.</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – PDF Upload
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### 📤 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file to analyse",
    type=["pdf"],
    help="Upload any PDF document such as an employee handbook, policy, manual, or report.",
    key="pdf_uploader",
)

if uploaded_file is not None:
    # ── Validate the uploaded file ──
    file_bytes = uploaded_file.getvalue()
    is_valid, error_msg = validate_pdf(file_bytes, uploaded_file.name)

    if not is_valid:
        st.markdown(f'<div class="status-error">❌ {error_msg}</div>', unsafe_allow_html=True)
    else:
        # ── Show file details & processing trigger ──
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f'<div class="doc-card">'
                f'<b>📎 Document:</b> {uploaded_file.name}<br>'
                f'<b>📦 File size:</b> {len(file_bytes) / 1024:.1f} KB'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            process_btn = st.button(
                "⚙️ Process PDF",
                use_container_width=True,
                type="primary",
                key="process_pdf_btn",
            )

        if process_btn:
            tmp_path = None
            try:
                st.session_state.processing = True
                with st.spinner("📖 Extracting text from PDF…"):
                    tmp_path = save_uploaded_file(uploaded_file)
                    time.sleep(0.3)  # Let UI update

                with st.spinner("✂️ Splitting into chunks…"):
                    time.sleep(0.2)

                with st.spinner("🔢 Generating embeddings and building ChromaDB…"):
                    vector_store, total_pages, total_chunks = process_pdf(str(tmp_path))

                # Persist results in session state
                st.session_state.vector_store = vector_store
                st.session_state.doc_info = {
                    "filename": uploaded_file.name,
                    "pages":    total_pages,
                    "chunks":   total_chunks,
                }
                st.session_state.chat_history = []  # Reset chat for new document

                st.markdown(
                    f'<div class="status-success">'
                    f'✅ <b>Successfully Processed!</b><br>'
                    f'📄 <b>Document:</b> {uploaded_file.name} &nbsp;|&nbsp; '
                    f'📃 <b>Pages:</b> {total_pages} &nbsp;|&nbsp; '
                    f'🧩 <b>Chunks:</b> {total_chunks}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.5)
                st.rerun()

            except FileNotFoundError as e:
                st.markdown(f'<div class="status-error">❌ File Error: {e}</div>', unsafe_allow_html=True)
            except ValueError as e:
                st.markdown(f'<div class="status-error">❌ PDF Content Error: {e}</div>', unsafe_allow_html=True)
            except Exception as e:
                err_str = str(e)
                if "connect" in err_str.lower() or "connection" in err_str.lower():
                    st.markdown(
                        '<div class="status-error">❌ Unable to connect to Ollama. '
                        'Please make sure Ollama is installed and running.<br>'
                        '<code>ollama serve</code></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="status-error">❌ Processing Error: {err_str}</div>',
                        unsafe_allow_html=True,
                    )
            finally:
                st.session_state.processing = False
                if tmp_path:
                    cleanup_temp_file(tmp_path)

        # ── Show currently processed doc info (if already processed) ──
        elif st.session_state.doc_info:
            info = st.session_state.doc_info
            st.markdown(
                f'<div class="status-success">'
                f'✅ <b>Document ready:</b> {info["filename"]} &nbsp;|&nbsp; '
                f'{info["pages"]} pages &nbsp;|&nbsp; {info["chunks"]} chunks'
                f'</div>',
                unsafe_allow_html=True,
            )


st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Chat Interface
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### 💬 Ask Questions")

# ── Render existing chat history ──
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-message">👤 <b>You:</b><br>{message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-message">🤖 <b>Assistant:</b><br>{message["content"]}</div>',
            unsafe_allow_html=True,
        )

        # ── Source citations ──
        if message.get("sources"):
            source_html = " ".join(
                f'<span class="source-badge">📖 Page {s["page"]}</span>'
                for s in message["sources"]
            )
            st.markdown(
                f'<div style="margin-top:0.4rem;"><b style="color:#8892b0;font-size:0.85rem;">Sources:</b> {source_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Collapsible retrieved context ──
        if message.get("chunks"):
            with st.expander(f"📂 View Retrieved Context (message {i // 2 + 1})", expanded=False):
                for j, chunk in enumerate(message["chunks"], 1):
                    page = chunk.metadata.get("page", "?")
                    preview = truncate_text(chunk.page_content, 400)
                    st.markdown(
                        f"**Chunk {j} — Page {page}**\n\n> {preview}",
                        unsafe_allow_html=False,
                    )
                    if j < len(message["chunks"]):
                        st.markdown("---")


# ── Question Input ──
question = st.chat_input(
    placeholder="Ask a question about your document…",
    key="chat_input",
)

if question:
    # Guard: document must be loaded first
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF document first before asking questions.")
    else:
        # Add user message to history immediately
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Generate the answer
        with st.spinner("🧠 Thinking… (this may take a few seconds with a local LLM)"):
            try:
                result = generate_answer(st.session_state.vector_store, question)

                st.session_state.chat_history.append(
                    {
                        "role":    "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "chunks":  result["chunks"],
                    }
                )

            except Exception as e:
                err_str = str(e)
                if "connect" in err_str.lower() or "connection refused" in err_str.lower():
                    error_content = (
                        "❌ Unable to connect to Ollama. "
                        "Please make sure Ollama is installed and running (`ollama serve`)."
                    )
                elif "model" in err_str.lower() and "not found" in err_str.lower():
                    error_content = (
                        f"❌ The model '{OLLAMA_MODEL_NAME}' is not installed in Ollama. "
                        f"Please run: `ollama pull {OLLAMA_MODEL_NAME}`"
                    )
                else:
                    error_content = f"❌ An error occurred while generating the answer: {err_str}"

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_content, "sources": [], "chunks": []}
                )

        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Footer
# ──────────────────────────────────────────────────────────────────────────────

if not st.session_state.chat_history and st.session_state.vector_store is None:
    st.markdown(
        """
        <div style="text-align:center;padding:2rem;color:#556;">
            <p style="font-size:3rem;margin:0">📄</p>
            <p style="font-size:1.1rem;font-weight:500;color:#8892b0;">
                Upload a PDF to get started
            </p>
            <p style="font-size:0.9rem;">
                Your document will be processed locally — no data leaves your machine.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.chat_history and st.session_state.vector_store is not None:
    st.markdown(
        """
        <div style="text-align:center;padding:2rem;color:#556;">
            <p style="font-size:3rem;margin:0">💬</p>
            <p style="font-size:1.1rem;font-weight:500;color:#8892b0;">
                Document ready! Ask your first question above.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
