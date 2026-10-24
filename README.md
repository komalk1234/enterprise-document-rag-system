# Enterprise Document RAG System

> **Final-Year BE Information Technology Mini-Project | AI & Data Science**

---

## Overview

The **Enterprise Document RAG System** is a fully offline, AI-powered document question-answering application. Upload any PDF — an employee handbook, company policy, user manual, or academic report — and ask questions in plain English. The system retrieves the most relevant passages from your document and uses a locally running open-source large language model (LLM) to generate accurate, grounded answers.

**No internet required after setup. No API keys. No cloud services.**

---

## Problem Statement

Organizations maintain large volumes of documents — HR policies, compliance manuals, product guides, and reports. Finding specific information manually is slow and error-prone. This project provides an AI-based question-answering system that:

1. Reads and indexes any uploaded PDF.
2. Understands natural-language questions.
3. Retrieves the most relevant passages using semantic search.
4. Generates precise answers grounded *only* in the uploaded document.
5. Cites the exact page numbers used to form the answer.

---

## Features

- 📤 **PDF Upload** – Drag-and-drop upload with validation
- 📄 **Page-preserving extraction** – Page numbers tracked throughout
- ✂️ **Smart chunking** – Overlapping text chunks for better retrieval
- 🔢 **Local embeddings** – HuggingFace all-MiniLM-L6-v2 (no API needed)
- 🗄️ **ChromaDB vector store** – Fast semantic search
- 🤖 **Local LLM** – Llama 3.2 3B via Ollama (fully offline)
- 💬 **Chat interface** – Multi-turn Q&A with history
- 📖 **Source citations** – Page numbers shown below every answer
- 📂 **View retrieved context** – Collapsible raw chunks for transparency
- 🚫 **Hallucination prevention** – LLM instructed to refuse out-of-context questions
- 🗑️ **Clear Chat** – Reset conversation any time
- 🌙 **Dark-mode UI** – Professional Streamlit interface
- ⚙️ **Ollama status checker** – One-click connectivity check in the sidebar

---

## Architecture

```
PDF Upload
     ↓
Text Extraction (PyPDFLoader – page-by-page)
     ↓
Text Cleaning (remove noise, whitespace)
     ↓
Text Chunking (RecursiveCharacterTextSplitter)
     ↓
Sentence Transformer Embeddings (all-MiniLM-L6-v2)
     ↓
ChromaDB (vector store on disk)
     ↓
User Question → Semantic Search → Relevant Context
     ↓
Anti-hallucination Prompt + Context → Llama 3.2 via Ollama
     ↓
Grounded Answer + Source Page Numbers
```

---

## Technologies Used

| Technology | Purpose | Why This Choice |
|---|---|---|
| **Python 3.10+** | Core language | Dominant ML/AI ecosystem |
| **Streamlit** | Web UI | Rapid prototyping, pure Python |
| **LangChain** | RAG orchestration | Modular pipelines, active community |
| **ChromaDB** | Vector database | Lightweight, embedded, no server needed |
| **Sentence Transformers** | Text embeddings | Local, high-quality, free |
| **Ollama** | Local LLM server | Run open-source LLMs on any laptop |
| **Llama 3.2 3B** | LLM | Best quality-to-size ratio for laptops |
| **PyPDF** | PDF parsing | Reliable text + metadata extraction |

---

## Chunking Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 1000 characters | Long enough to preserve context; short enough for precise retrieval and LLM context limits |
| `chunk_overlap` | 200 characters | Prevents key sentences that fall on chunk boundaries from being lost |

---

## Installation (Windows)

### Prerequisites

- Python 3.10 or higher
- Git (optional)
- [Ollama](https://ollama.com) installed

### Step 1 – Clone / Download the project

```bash
cd enterprise-rag
```

### Step 2 – Create a virtual environment

```bash
python -m venv venv
```

### Step 3 – Activate the virtual environment

```bash
.\venv\Scripts\activate
```

You should see `(venv)` at the beginning of your prompt.

### Step 4 – Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This may take a few minutes. The Sentence Transformer model (~80 MB) is downloaded automatically on first run.

### Step 5 – Install Ollama

Download and install Ollama from: https://ollama.com/download/windows

After installation, **restart your terminal** so that the `ollama` command is recognized.

### Step 6 – Pull the Llama model

```bash
ollama pull llama3.2:3b
```

> ⚠️ This downloads ~2 GB. Requires internet. Only needed once.

### Step 7 – Verify the model works

```bash
ollama run llama3.2:3b
```

Type anything, press Enter, then `Ctrl+D` or `/bye` to exit.

### Step 8 – Run the application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

> **Note:** Ollama runs automatically in the background on Windows after installation. If answers fail, open a separate terminal and run `ollama serve` to start the server manually.

---

## How to Use

1. **Start Ollama** – It usually starts automatically on Windows. If not: `ollama serve`
2. **Start the app** – `streamlit run app.py`
3. **Upload a PDF** – Click the file uploader and select your PDF
4. **Click "⚙️ Process PDF"** – Wait for the progress spinners to complete
5. **Ask questions** – Type in the chat input at the bottom
6. **View answers** – See the AI answer with source page numbers
7. **View retrieved context** – Expand the "📂 View Retrieved Context" section
8. **Clear chat** – Use the "🗑️ Clear Chat" button in the sidebar

---

## Sample Document

See `sample_documents/technova_handbook.md` — the TechNova Solutions Employee Handbook.

**Convert to PDF for use:**

Option A – Pandoc (if installed):
```bash
pandoc sample_documents/technova_handbook.md -o technova_handbook.pdf
```

Option B – Paste into Google Docs → File → Download → PDF

Option C – Use https://md2pdf.netlify.app/

---

## Demo Questions

Use the **TechNova Solutions Employee Handbook** PDF for a reliable demo.

| # | Question | Expected Answer |
|---|---|---|
| 1 | How many paid annual leave days are employees entitled to? | 20 days |
| 2 | What are the company's working hours? | 9:00 AM to 6:00 PM, Monday to Friday |
| 3 | Can employees work from home? | Up to 2 days per week with manager approval |
| 4 | What is the notice period? | 60 days |
| 5 | Who is covered under health insurance? | Employee and immediate family (spouse + up to 2 dependent children) |
| 6 | What is the probation period? | Six months |

**Out-of-context test (hallucination prevention):**

> **Question:** "Who is the President of the United States?"
>
> **Expected:** "I couldn't find this information in the uploaded document."

---

## RAG Explanation

**Retrieval-Augmented Generation (RAG)** is a technique where an AI system first *retrieves* relevant passages from a knowledge source (your PDF), then *generates* an answer using only those passages. This is different from a regular chatbot that answers from its training data. RAG grounds the LLM in your specific document, preventing hallucination and making the answers accurate and auditable.

---

## Embeddings Explanation

An **embedding** is a list of numbers (a vector) that represents the meaning of a piece of text. Similar sentences produce similar vectors. When you ask a question, it is converted into an embedding, and the system finds the document chunks whose embeddings are closest — i.e., most semantically similar — to your question's embedding.

---

## ChromaDB Explanation

**ChromaDB** is a vector database — a database specifically designed to store and search embeddings. When you process a PDF, all chunk embeddings are stored in ChromaDB on your disk. When you ask a question, ChromaDB performs a fast similarity search and returns the most relevant chunks in milliseconds.

---

## Limitations

- Scanned / image-only PDFs will not work — they contain no extractable text. Use OCR (e.g., Adobe Acrobat, Tesseract) to pre-process such PDFs.
- Answer quality depends on the quality of text extracted from the PDF.
- A 3B-parameter local LLM is less powerful than larger cloud models; complex reasoning questions may produce weaker answers.
- The first run requires an internet connection to download the embedding model (~80 MB) and the Llama model (~2 GB).
- After setup, the system works fully offline.

---

## Future Scope

- Multiple PDFs in one session
- DOCX, TXT, and spreadsheet support
- OCR integration for scanned documents
- Multilingual document support
- Larger local LLMs (Llama 3 8B, Mistral 7B)
- Role-based enterprise access control
- Persistent document libraries with named collections
- Re-ranking of retrieved chunks for improved accuracy

---

## Viva Quick Notes

**1. What is RAG?**
RAG (Retrieval-Augmented Generation) is a technique where we first retrieve relevant text passages from a knowledge source, then feed those passages to an LLM to generate a grounded answer. It combines information retrieval with text generation.

**2. Why did you use RAG?**
A standard LLM only knows what it was trained on. RAG allows the model to answer questions about any custom document — an uploaded PDF — without retraining, and it prevents the model from making up information that isn't in the document.

**3. What is an embedding?**
An embedding is a numerical vector (list of numbers) that represents the semantic meaning of a text. Similar texts produce similar vectors. They allow us to do mathematical comparisons of meaning — something impossible with raw text.

**4. What is a vector database?**
A vector database stores text embeddings and enables fast nearest-neighbor search — finding the stored vectors most similar to a query vector. It is the backbone of semantic search in RAG systems.

**5. Why ChromaDB?**
ChromaDB is lightweight, open-source, embedded (no separate server), and integrates directly with LangChain. It stores data locally on disk — ideal for a student project without infrastructure.

**6. Why LangChain?**
LangChain provides ready-made, modular components for building RAG pipelines — document loaders, text splitters, vector store wrappers, prompt templates, and LLM connectors. It dramatically reduces boilerplate code.

**7. Why Ollama?**
Ollama makes it trivial to download and run open-source LLMs locally on a laptop using a simple CLI. It serves the model via a local REST API, so no internet connection is needed during use.

**8. Why Llama 3.2 3B?**
Llama 3.2 3B is Meta's open-source model tuned for instruction-following. At 3 billion parameters it fits comfortably in a typical laptop's RAM (~3-4 GB) while still producing coherent, accurate responses.

**9. What is semantic search?**
Semantic search finds results based on *meaning* rather than exact keyword matching. A question like "How many vacation days?" will still match text that says "annual leave entitlement" because their embeddings are close in vector space.

**10. What is chunking?**
Chunking is splitting a large document into smaller, overlapping segments before embedding. LLMs have limited context windows, and smaller chunks are also more precise retrieval units — only the relevant section is sent to the LLM, not the entire document.

**11. Why do we use chunk overlap?**
If a sentence falls exactly at a chunk boundary, it would be split across two chunks and might be incomplete in either. Overlap ensures every sentence appears fully in at least one chunk, preventing information loss.

**12. Difference between normal chatbot and RAG chatbot.**
A normal chatbot answers from its training data — it cannot access your private documents and often hallucinates. A RAG chatbot retrieves information from your specific document first, then generates an answer grounded in that content — making it accurate, auditable, and document-specific.

**13. What happens when a user asks a question?**
The question is converted into an embedding → ChromaDB finds the top-k most similar chunks → those chunks are combined into a context block → the context + question are inserted into an anti-hallucination prompt → the prompt is sent to Llama 3.2 via Ollama → the model generates an answer based only on the provided context.

**14. How does your system reduce hallucination?**
The LLM is given an explicit prompt instruction: "Answer using only the provided document context. If the answer is not in the context, say so." This forces the model to stay within the retrieved information rather than using its general training knowledge.

**15. Why is this useful for enterprises?**
Enterprises have thousands of pages of internal documents — HR policies, compliance guidelines, technical manuals. Employees waste hours searching for specific information. An RAG system makes this instant, accurate, and trustworthy, while keeping sensitive documents private on local infrastructure.

---

*Built with ❤️ as a Final-Year BE Information Technology Mini-Project*
