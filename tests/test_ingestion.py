"""
Basic tests for the ingestion pipeline.
Run with: uv run pytest tests/
"""

import pytest
from pathlib import Path


# ── Chunker tests ─────────────────────────────────────────────────────────────

def test_detect_section_skills():
    from resume_bot.ingestion.chunker import detect_section
    assert detect_section("Python, JavaScript, React, Node.js, Skills") == "skills"


def test_detect_section_experience():
    from resume_bot.ingestion.chunker import detect_section
    assert detect_section("Work Experience at Google — 2022-2024") == "experience"


def test_detect_section_education():
    from resume_bot.ingestion.chunker import detect_section
    assert detect_section("B.Tech Computer Science, University of Delhi") == "education"


def test_detect_section_general():
    from resume_bot.ingestion.chunker import detect_section
    assert detect_section("Some random text that matches nothing") == "general"


def test_chunk_summary_structure():
    from resume_bot.ingestion.chunker import get_chunk_summary
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Python developer with 3 years experience",
            metadata={"section": "experience", "char_count": 40}
        ),
        Document(
            page_content="Skills: Python, LangChain, FastAPI",
            metadata={"section": "skills", "char_count": 35}
        ),
    ]
    summary = get_chunk_summary(docs)
    assert "total_chunks" in summary
    assert "section_distribution" in summary
    assert "avg_chunk_size" in summary
    assert summary["total_chunks"] == 2


# ── Config tests ──────────────────────────────────────────────────────────────

def test_config_chunk_size():
    from resume_bot.config import CHUNK_SIZE, CHUNK_OVERLAP
    assert 300 <= CHUNK_SIZE <= 700, "Chunk size should be tuned for resume pages"
    assert CHUNK_OVERLAP < CHUNK_SIZE, "Overlap must be smaller than chunk size"


def test_config_retriever_params():
    from resume_bot.config import RETRIEVER_K, RETRIEVER_FETCH_K
    assert RETRIEVER_FETCH_K >= RETRIEVER_K, "fetch_k must be >= k for MMR to work"


def test_embedding_dimensions():
    from resume_bot.config import EMBEDDING_DIMENSIONS
    assert EMBEDDING_DIMENSIONS == 768, "text-embedding-004 uses 768 dimensions"
