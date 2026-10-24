"""
rag_pipeline.py - Core RAG (Retrieval-Augmented Generation) Pipeline

This module handles the entire RAG workflow:
1. Loading PDF and extracting text page-by-page
2. Splitting text into overlapping chunks
3. Creating embeddings with a local Sentence Transformer
4. Storing chunks + embeddings in ChromaDB
5. Retrieving relevant chunks for a user question
6. Sending context + question to Ollama LLM
7. Returning the grounded answer + source page metadata
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*")

import os
import shutil
from pathlib import Path
from typing import Optional

# LangChain document loaders and splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings via local Sentence Transformers (no API key needed)
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB vector store integration
from langchain_chroma import Chroma

# Ollama LLM integration
from langchain_ollama import OllamaLLM

# LangChain core for prompt templates and chain construction
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from utils import clean_text

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# The local embedding model – runs 100 % offline after first download
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The local LLM served by Ollama
OLLAMA_MODEL_NAME = "llama3.2:3b"
OLLAMA_BASE_URL   = "http://localhost:11434"

# Chunking parameters (tuned for enterprise documents)
# chunk_size = 1000 chars keeps each chunk meaningful but short enough for the
# LLM context window.  overlap = 200 chars prevents key sentences from being
# cut at chunk boundaries.
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Number of chunks retrieved per query
TOP_K_RESULTS = 4

# Directory for the ChromaDB persistent store
CHROMA_DB_DIR        = Path("chroma_db")
CHROMA_COLLECTION_NAME = "enterprise_rag"

# ──────────────────────────────────────────────────────────────────────────────
# ANTI-HALLUCINATION PROMPT TEMPLATE
# The LLM is explicitly told to use ONLY the provided context and to decline
# gracefully when the answer is not present.
# ──────────────────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are an enterprise document assistant. Your job is to answer the user's question using ONLY the information provided in the document context below.

Rules:
- Answer based solely on the context provided.
- Do NOT use any external knowledge or make up information.
- If the answer is not found in the context, respond with exactly: "I couldn't find this information in the uploaded document."
- Be concise and clear.
- Where relevant, mention which part of the document supports your answer.

Document Context:
{context}

User Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. PDF LOADING & TEXT EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> tuple[list[Document], int]:
    """
    Load a PDF file and extract text page-by-page using PyPDFLoader.

    Each returned Document contains:
      - page_content : text of that page (after basic cleaning)
      - metadata     : {'source': filename, 'page': 1-based page number}

    Args:
        pdf_path: Absolute path to the PDF file on disk.

    Returns:
        (documents, total_pages)

    Raises:
        ValueError: If the PDF has no extractable text.
        FileNotFoundError: If the file path does not exist.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    # PyPDFLoader extracts text page-by-page and adds metadata automatically
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError("The PDF appears to be empty or could not be parsed.")

    # Clean the text on each page and make page numbers 1-based
    cleaned_docs = []
    for doc in raw_docs:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:  # Skip completely blank pages
            # PyPDFLoader uses 0-based page index in metadata; convert to 1-based
            page_num = doc.metadata.get("page", 0) + 1
            cleaned_docs.append(
                Document(
                    page_content=cleaned_content,
                    metadata={
                        "source": Path(pdf_path).name,
                        "page":   page_num,
                    },
                )
            )

    total_pages = len(raw_docs)

    if not cleaned_docs:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image-only PDF. "
            "Please use a PDF with selectable text."
        )

    return cleaned_docs, total_pages


# ──────────────────────────────────────────────────────────────────────────────
# 2. TEXT CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split each page document into smaller overlapping chunks.

    Why chunk?  LLMs have limited context windows.  Smaller, focused chunks
    also improve retrieval precision – only the most relevant passage is
    fetched, not the entire page.

    Why overlap?  Important sentences that fall near a chunk boundary would
    otherwise be cut in half.  Overlap ensures those sentences appear fully
    in at least one chunk.

    Args:
        documents: List of page-level Document objects.

    Returns:
        List of chunk-level Document objects, each retaining 'source' and
        'page' metadata from its parent page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph, then sentence, then word boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDINGS SETUP
# ──────────────────────────────────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace Sentence Transformer embedding model.

    all-MiniLM-L6-v2 is a lightweight 80 MB model that produces high-quality
    384-dimensional embeddings for semantic similarity tasks.
    It runs entirely on CPU, making it suitable for student laptops.

    Returns:
        HuggingFaceEmbeddings instance ready to encode text.

    Raises:
        Exception: If the model cannot be loaded (e.g., no internet for first
                   download, or corrupted cache).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # Use CPU (safe for any laptop)
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# 4. CHROMADB VECTOR STORE
# ──────────────────────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> Chroma:
    """
    Create (or rebuild) a ChromaDB vector store from the given document chunks.

    The existing collection is DELETED before rebuilding so that content from
    a previously uploaded PDF never contaminates answers for a new PDF.

    Args:
        chunks:      List of Document chunks (with metadata).

    Returns:
        Chroma vector store instance ready for similarity search.
    """
    # ── Wipe old data so previous PDF doesn't leak into new queries ──
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    embeddings = get_embedding_model()

    # Chroma.from_documents() encodes all chunks and persists to disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )

    return vector_store


def load_vector_store() -> Optional[Chroma]:
    """
    Load an existing ChromaDB vector store from disk (if one exists).

    Returns:
        Chroma instance, or None if no store has been built yet.
    """
    if not CHROMA_DB_DIR.exists():
        return None

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    return vector_store


# ──────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    vector_store: Chroma,
    question: str,
    k: int = TOP_K_RESULTS,
) -> list[Document]:
    """
    Perform semantic (vector) similarity search against ChromaDB.

    The question is embedded with the same model used for indexing, then
    ChromaDB finds the k chunks whose embeddings are closest (cosine
    similarity) to the question embedding.

    Args:
        vector_store: Populated Chroma vector store.
        question:     The user's natural-language question.
        k:            Number of top chunks to retrieve.

    Returns:
        List of the most relevant Document chunks.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    relevant_docs = retriever.invoke(question)
    return relevant_docs


# ──────────────────────────────────────────────────────────────────────────────
# 6. OLLAMA LLM SETUP
# ──────────────────────────────────────────────────────────────────────────────

def get_llm() -> OllamaLLM:
    """
    Create an OllamaLLM instance pointing at the local Ollama server.

    Ollama must be running (ollama serve) and llama3.2:3b must be pulled.

    Returns:
        OllamaLLM instance.
    """
    llm = OllamaLLM(
        model=OLLAMA_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,   # Low temperature → more factual / less creative
    )
    return llm


# ──────────────────────────────────────────────────────────────────────────────
# 7. ANSWER GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(
    vector_store: Chroma,
    question: str,
) -> dict:
    """
    Full RAG pipeline: retrieve → format context → prompt LLM → return answer.

    Steps:
      1. Retrieve the top-k most relevant chunks from ChromaDB.
      2. Concatenate their text into a single context block.
      3. Fill the anti-hallucination prompt template.
      4. Send to Ollama and get a grounded answer.
      5. Return the answer, source pages, and raw chunks for display.

    Args:
        vector_store: Populated Chroma vector store.
        question:     User's natural-language question.

    Returns:
        dict with keys:
          - "answer"   : str  – the LLM's grounded answer
          - "sources"  : list[dict] – [{"page": int, "source": str}, ...]
          - "chunks"   : list[Document] – raw retrieved chunks (for "View Context")
    """
    # Step 1 – Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(vector_store, question)

    if not relevant_chunks:
        return {
            "answer":  "I couldn't find this information in the uploaded document.",
            "sources": [],
            "chunks":  [],
        }

    # Step 2 – Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        page = chunk.metadata.get("page", "?")
        context_parts.append(f"[Chunk {i} – Page {page}]\n{chunk.page_content}")

    context = "\n\n".join(context_parts)

    # Step 3 – Fill the prompt template
    filled_prompt = RAG_PROMPT.format(context=context, question=question)

    # Step 4 – Call the local LLM via Ollama
    llm    = get_llm()
    answer = llm.invoke(filled_prompt)

    # Step 5 – Collect unique source metadata for citation display
    seen_pages = set()
    sources    = []
    for chunk in relevant_chunks:
        page   = chunk.metadata.get("page",   "?")
        source = chunk.metadata.get("source", "unknown")
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "source": source})

    # Sort by page number (numeric sort where possible)
    sources.sort(key=lambda x: int(x["page"]) if str(x["page"]).isdigit() else 0)

    return {
        "answer":  answer.strip(),
        "sources": sources,
        "chunks":  relevant_chunks,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. FULL DOCUMENT PROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str) -> tuple[Chroma, int, int]:
    """
    High-level convenience function: load → chunk → embed → store.

    Call this after a user uploads a new PDF.

    Args:
        pdf_path: Absolute path to the saved PDF file.

    Returns:
        (vector_store, total_pages, total_chunks)

    Raises:
        FileNotFoundError: PDF path is wrong.
        ValueError:        PDF has no extractable text.
        Exception:         Embedding or ChromaDB error.
    """
    # Load and clean pages
    documents, total_pages = load_pdf(pdf_path)

    # Split into overlapping chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    # Build (or rebuild) the vector store
    vector_store = build_vector_store(chunks)

    return vector_store, total_pages, len(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# QUICK CONNECTIVITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

def check_ollama_connection() -> tuple[bool, str]:
    """
    Try a trivial call to Ollama to verify it is reachable.

    Returns:
        (ok: bool, message: str)
    """
    try:
        import httpx
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data   = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if any(OLLAMA_MODEL_NAME in m for m in models):
                return True, f"Ollama is running. Model '{OLLAMA_MODEL_NAME}' is available."
            else:
                return False, (
                    f"Ollama is running but the model '{OLLAMA_MODEL_NAME}' is not installed.\n"
                    f"Please run: ollama pull {OLLAMA_MODEL_NAME}"
                )
        return False, "Ollama returned an unexpected response."
    except Exception as exc:
        return False, (
            "Unable to connect to Ollama. "
            "Please make sure Ollama is installed and running.\n"
            "Start it with: ollama serve"
        )
