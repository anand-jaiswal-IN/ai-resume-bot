"""
PDF Resume Loader — uses PyMuPDF for robust text and hyperlink extraction.

Why PyMuPDFLoader over PyPDFLoader?
  - Preserves more layout and formatting structure
  - Extracts embedded hyperlinks (LinkedIn, GitHub, portfolio URLs)
  - Handles multi-column resume layouts better
  - Faster parsing
"""

from pathlib import Path
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF modern import
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from resume_bot.config import DATA_DIR


@dataclass
class ResumeMetadata:
    """Structured metadata extracted from the resume PDF."""

    file_name: str
    total_pages: int
    links: list[str] = field(default_factory=list)
    candidate_name: str = "Unknown"


def extract_links(pdf_path: str | Path) -> list[str]:
    """
    Extract all hyperlinks from the PDF using PyMuPDF's annotation API.
    Returns a deduplicated list of URIs found in the document.
    """
    pdf_path = Path(pdf_path)
    links: list[str] = []

    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri and uri not in links:
                    links.append(uri)

    return links


def load_resume(pdf_path: str | Path) -> tuple[list[Document], ResumeMetadata]:
    """
    Load a resume PDF and return LangChain Documents + extracted metadata.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A tuple of:
          - List of LangChain Document objects (one per page)
          - ResumeMetadata dataclass with links and file info
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume not found at: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.suffix}")

    # Extract hyperlinks before loading (separate PyMuPDF pass)
    links = extract_links(pdf_path)

    # Load pages as LangChain Documents via PyMuPDFLoader
    loader = PyMuPDFLoader(str(pdf_path))
    documents: list[Document] = loader.load()

    # Enrich every document page's metadata
    for doc in documents:
        doc.metadata["source_file"] = pdf_path.name
        # ChromaDB requires metadata values to be str/int/float/bool — serialize list
        doc.metadata["links"] = ", ".join(links) if links else ""
        doc.metadata["total_pages"] = len(documents)

    metadata = ResumeMetadata(
        file_name=pdf_path.name,
        total_pages=len(documents),
        links=links,
    )

    return documents, metadata


def resolve_resume_path(user_input: str) -> Path:
    """
    Resolve a user-supplied path string to an absolute Path.
    If only a filename is given, looks inside DATA_DIR.
    """
    path = Path(user_input)

    if path.is_absolute() and path.exists():
        return path

    # Try relative to cwd
    if path.exists():
        return path.resolve()

    # Try inside data/resumes/
    candidate = DATA_DIR / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot find '{user_input}'. "
        f"Place your PDF inside '{DATA_DIR}' or provide the full path."
    )
