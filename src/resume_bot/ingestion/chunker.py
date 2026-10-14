"""
Section-aware resume text chunker.

Strategy
────────
1. Use RecursiveCharacterTextSplitter with resume-specific separators so
   job entries, bullet points, and skill blocks stay intact.
2. Inject section-label metadata derived from keyword detection in chunk
   content — enables future metadata-filtered retrieval.
3. Deduplicate near-identical chunks (can happen on 2-page resumes with
   repeated headers).
"""

import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from resume_bot.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
)

# ── Section detection keywords ────────────────────────────────────────────────
# Maps a canonical section name → keywords to look for (case-insensitive)
SECTION_KEYWORDS: dict[str, list[str]] = {
    "summary": ["summary", "objective", "profile", "about me", "overview"],
    "experience": ["experience", "work history", "employment", "career", "internship"],
    "education": ["education", "academic", "degree", "university", "college", "school", "certification"],
    "skills": ["skills", "technologies", "tech stack", "tools", "languages", "frameworks", "competencies"],
    "projects": ["projects", "portfolio", "personal projects", "open source", "side projects"],
    "achievements": ["achievements", "awards", "honors", "recognition", "accomplishments"],
    "publications": ["publications", "papers", "research", "thesis"],
    "contact": ["contact", "email", "phone", "linkedin", "github", "address", "location"],
    "certifications": ["certifications", "certificates", "credential", "license"],
    "languages": ["languages spoken", "spoken languages"],
}


def detect_section(text: str) -> str:
    """
    Heuristically detect which resume section a text chunk belongs to.
    Returns the section name or 'general' if no match found.
    """
    text_lower = text.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return section
    return "general"


def _content_hash(text: str) -> str:
    """Short SHA-256 fingerprint for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split resume pages into semantically coherent chunks.

    - Applies RecursiveCharacterTextSplitter with resume-tuned parameters.
    - Enriches each chunk with section metadata.
    - Removes near-duplicate chunks via content hash.

    Args:
        documents: List of LangChain Documents (one per PDF page).

    Returns:
        Deduplicated list of enriched Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,       # Preserve bullet chars for context
        add_start_index=True,      # Track char offset for debugging
    )

    raw_chunks: list[Document] = splitter.split_documents(documents)

    seen_hashes: set[str] = set()
    enriched_chunks: list[Document] = []

    for i, chunk in enumerate(raw_chunks):
        # Skip near-duplicate chunks
        h = _content_hash(chunk.page_content)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Detect section for metadata-filtered retrieval
        section = detect_section(chunk.page_content)

        chunk.metadata.update(
            {
                "chunk_index": i,
                "section": section,
                "content_hash": h,
                "char_count": len(chunk.page_content),
            }
        )

        enriched_chunks.append(chunk)

    return enriched_chunks


def get_chunk_summary(chunks: list[Document]) -> dict:
    """Return a summary of the chunking result for display."""
    sections: dict[str, int] = {}
    for chunk in chunks:
        section = chunk.metadata.get("section", "general")
        sections[section] = sections.get(section, 0) + 1

    return {
        "total_chunks": len(chunks),
        "section_distribution": sections,
        "avg_chunk_size": (
            sum(c.metadata.get("char_count", 0) for c in chunks) // max(len(chunks), 1)
        ),
    }
