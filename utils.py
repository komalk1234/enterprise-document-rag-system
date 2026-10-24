"""
utils.py - Helper utilities for the Enterprise RAG System

Contains small, reusable functions for:
- Text cleaning
- PDF validation
- Page number formatting
- Temp file handling
"""

import re
import os
import tempfile
import shutil
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing excessive whitespace
    and non-printable characters.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Replace multiple whitespace/newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-printable characters (keep normal punctuation)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)

    return text.strip()


def format_page_numbers(pages: list) -> str:
    """
    Format a list of page numbers into a readable string.
    Removes duplicates and sorts the list.

    Example: [3, 1, 3, 2] → "Page 1, Page 2, Page 3"

    Args:
        pages: List of page numbers (integers or strings).

    Returns:
        Formatted string of unique, sorted page numbers.
    """
    if not pages:
        return "Unknown"

    # Convert to int where possible, remove duplicates, sort
    unique_pages = sorted(set(int(p) for p in pages if str(p).isdigit()))

    if not unique_pages:
        return "Unknown"

    return ", ".join(f"Page {p}" for p in unique_pages)


def validate_pdf(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """
    Validate that the uploaded file looks like a real PDF.

    Args:
        file_bytes: The raw bytes of the uploaded file.
        filename:   The original filename from the uploader.

    Returns:
        (is_valid: bool, error_message: str)
        error_message is empty string if valid.
    """
    # Check file extension
    if not filename.lower().endswith(".pdf"):
        return False, "Uploaded file does not have a .pdf extension."

    # Check PDF magic bytes (%PDF-)
    if not file_bytes.startswith(b"%PDF"):
        return False, "Uploaded file does not appear to be a valid PDF."

    # Basic size check (too small to be a real PDF)
    if len(file_bytes) < 100:
        return False, "Uploaded file is too small to be a valid PDF."

    return True, ""


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save a Streamlit UploadedFile object to a temporary file on disk
    and return the Path to it.

    Ollama/LangChain loaders need a real file path, not a BytesIO buffer.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        pathlib.Path pointing to the temporary file.
    """
    # Create a temp directory that persists until explicitly deleted
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / uploaded_file.name

    # Write bytes to disk
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return tmp_path


def cleanup_temp_file(tmp_path: Path):
    """
    Delete a temporary file and its parent directory created by
    save_uploaded_file().

    Args:
        tmp_path: Path to the temp file.
    """
    try:
        if tmp_path and tmp_path.exists():
            shutil.rmtree(tmp_path.parent, ignore_errors=True)
    except Exception:
        pass  # Non-critical; OS will clean up temp folders eventually


def truncate_text(text: str, max_chars: int = 300) -> str:
    """
    Truncate a text snippet to max_chars for display purposes.

    Args:
        text:      Full text.
        max_chars: Maximum characters to show.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
