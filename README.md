# 🤖 AI Resume Bot

Interactive Context-Aware AI Resume Bot — talk to your PDF resume using RAG.

Built with **LangChain + Gemini + ChromaDB** for the CLI phase.

---

## Features

- 📄 **PDF ingestion** — PyMuPDF for rich text + hyperlink extraction
- 🧩 **Section-aware chunking** — understands Skills, Experience, Education, etc.
- 🔍 **MMR retrieval** — diverse, non-redundant context for the LLM
- 💬 **Multi-turn chat** — remembers conversation history per session
- ⚡ **Streaming** — token-by-token live output
- 🗃️ **ChromaDB** — persistent local vector store (SQLite-backed)

---

## Quick Start

### 1. Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

### 2. Setup

```bash
# Clone the repo
git clone <repo-url>
cd ai-resume-bot

# Install dependencies
uv sync

# Set up your API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run

```bash
# Load a resume and start chatting
uv run main.py data/resumes/your_resume.pdf

# Or start without a resume and load via CLI command
uv run main.py
```

---

## CLI Commands

| Command | Description |
|---|---|
| `/load <path>` | Load a new resume PDF |
| `/clear` | Clear conversation history |
| `/history` | Show chat history |
| `/links` | Show extracted hyperlinks |
| `/stats` | Show vector store stats |
| `/help` | Show all commands |
| `/exit` | Exit the app |

---

## Architecture

```
Resume PDF
    ↓ PyMuPDFLoader (text + links)
Section-Aware Chunking (chunk_size=500, overlap=75)
    ↓ Metadata injection (section labels)
Google text-embedding-004 (768 dimensions)
    ↓ ChromaDB (persistent SQLite)
MMR Retriever (k=5, fetch_k=10, λ=0.6)
    ↓ LCEL RAG Chain
Gemini 2.0 Flash (temp=0.1)
    ↓ Streaming output
Rich CLI
```

---

## RAG Parameters

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 500 | Full job entry or skill block |
| `chunk_overlap` | 75 | 15% overlap, prevents boundary loss |
| `embedding_model` | `text-embedding-004` | 768-dim Google native embedding |
| `retriever_k` | 5 | Optimal for 1-2 page resume |
| `search_type` | MMR | Avoids redundant skill chunks |
| `temperature` | 0.1 | Near-deterministic factual accuracy |

---

## Roadmap

- [x] **Phase 1** — CLI app (current)
- [ ] **Phase 2** — Enhanced RAG (contextual compression, HyDE)
- [ ] **Phase 3** — Frontend UI (FastAPI + Next.js/React)
- [ ] **Phase 4** — Production (FAISS/Pinecone, Docker, auth)

---

## Tech Stack

- **LLM**: `gemini-2.0-flash` via `langchain-google-genai`
- **Embeddings**: `text-embedding-004`
- **Vector DB**: ChromaDB → FAISS (production)
- **Framework**: LangChain LCEL
- **PDF**: PyMuPDF
- **CLI**: Rich
