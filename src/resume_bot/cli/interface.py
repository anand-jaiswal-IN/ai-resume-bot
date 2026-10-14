"""
Rich CLI interface for the AI Resume Bot.

Features:
  - Colorful, structured layout using the `rich` library
  - Streaming token output (no waiting for full response)
  - Multi-turn conversational mode with /commands
  - Automatic PDF ingestion on first run or when a new PDF is provided
  - Clear session, source display, and help commands
"""

import sys
from pathlib import Path
from typing import Callable, Iterator

from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.table import Table
from rich.text import Text

from resume_bot.config import APP_NAME, WELCOME_BANNER
from resume_bot.ingestion.loader import load_resume, resolve_resume_path, ResumeMetadata
from resume_bot.ingestion.chunker import chunk_documents, get_chunk_summary
from resume_bot.vectorstore.store import (
    index_documents,
    get_retriever,
    collection_is_populated,
    get_collection_stats,
)
from resume_bot.rag.chain import (
    build_conversational_chain,
    stream_conversational_answer,
    clear_session,
    get_session_history_messages,
)

console = Console()

# ── CLI Commands ──────────────────────────────────────────────────────────────
COMMANDS: dict[str, str] = {
    "/help":    "Show this help message",
    "/load":    "Load a new resume PDF  (e.g. /load resume.pdf)",
    "/clear":   "Clear the conversation history",
    "/history": "Show the current conversation history",
    "/links":   "Show all links extracted from the resume",
    "/stats":   "Show vector store statistics",
    "/exit":    "Exit the application",
    "/quit":    "Exit the application",
}


# ── Display helpers ───────────────────────────────────────────────────────────

def print_welcome() -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{APP_NAME}[/bold cyan]\n"
            "[dim]Powered by Gemini + LangChain + ChromaDB[/dim]\n\n"
            "[green]Upload your resume and ask anything about it.[/green]\n"
            "[dim]Type [bold]/help[/bold] to see available commands.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


def print_help() -> None:
    table = Table(title="Available Commands", border_style="dim", show_header=True)
    table.add_column("Command", style="bold yellow")
    table.add_column("Description", style="white")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


def print_metadata(meta: ResumeMetadata, chunk_summary: dict) -> None:
    table = Table(title="📄 Resume Loaded", border_style="green", show_header=False)
    table.add_column("Field", style="bold green")
    table.add_column("Value", style="white")

    table.add_row("File", meta.file_name)
    table.add_row("Pages", str(meta.total_pages))
    table.add_row("Links Found", str(len(meta.links)))
    table.add_row("Chunks Created", str(chunk_summary["total_chunks"]))
    table.add_row("Avg Chunk Size", f"{chunk_summary['avg_chunk_size']} chars")

    for section, count in chunk_summary["section_distribution"].items():
        table.add_row(f"  [{section}]", f"{count} chunk(s)")

    console.print(table)
    console.print()


def print_links(links: list[str]) -> None:
    if not links:
        console.print("[dim]No links found in the resume.[/dim]")
        return

    console.print(Panel(
        "\n".join(f"[link={url}]{url}[/link]" for url in links),
        title="🔗 Resume Links",
        border_style="blue",
    ))
    console.print()


def print_stats() -> None:
    stats = get_collection_stats()
    console.print(
        Panel(
            f"[bold]Collection:[/bold] {stats['collection']}\n"
            f"[bold]Indexed Chunks:[/bold] {stats['document_count']}",
            title="📊 Vector Store Stats",
            border_style="magenta",
        )
    )
    console.print()


def stream_and_print_answer(
    stream_fn: Callable[[], Iterator[str]],
) -> str:
    """
    Stream tokens from the LLM and print them live.
    Returns the full assembled answer string.
    """
    console.print()
    console.print(Text("🤖 Assistant", style="bold green"))
    console.print(Rule(style="dim green"))

    full_answer = ""
    with console.status("", spinner="dots"):
        pass  # spinner ends immediately; streaming replaces it

    # Streaming print — tokens appear one by one
    for token in stream_fn():
        console.print(token, end="", highlight=False)
        full_answer += token

    console.print()  # Newline after streaming ends
    console.print(Rule(style="dim green"))
    console.print()

    return full_answer


# ── Ingestion pipeline ────────────────────────────────────────────────────────

def ingest_resume(pdf_path: str | Path, reset: bool = True) -> ResumeMetadata:
    """
    Run the full ingestion pipeline for a resume PDF:
      load → chunk → embed → store

    Args:
        pdf_path: Path to the PDF file.
        reset:    Whether to clear the existing vector store first.

    Returns:
        ResumeMetadata for display.
    """
    with Status("[cyan]Loading PDF...[/cyan]", console=console):
        documents, meta = load_resume(pdf_path)

    with Status("[cyan]Chunking text...[/cyan]", console=console):
        chunks = chunk_documents(documents)
        chunk_summary = get_chunk_summary(chunks)

    with Status("[cyan]Embedding & indexing into ChromaDB...[/cyan]", console=console):
        store = index_documents(chunks, reset=reset)

    print_metadata(meta, chunk_summary)
    return meta


# ── Main CLI loop ─────────────────────────────────────────────────────────────

def run_cli(pdf_path: str | None = None) -> None:
    """
    Main interactive CLI loop.

    Args:
        pdf_path: Optional path to a resume PDF to load at startup.
    """
    print_welcome()

    current_links: list[str] = []
    SESSION_ID = "resume-chat"

    # ── Load resume on startup if provided ───────────────────────────────────
    if pdf_path:
        try:
            resolved = resolve_resume_path(pdf_path)
            meta = ingest_resume(resolved, reset=True)
            current_links = meta.links
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]❌ {e}[/red]")
            console.print("[dim]Use /load <path> to load a resume.[/dim]")
    elif collection_is_populated():
        stats = get_collection_stats()
        console.print(
            f"[green]✅ Using existing vector store "
            f"({stats['document_count']} chunks indexed).[/green]\n"
            "[dim]Use /load <path> to load a new resume.[/dim]\n"
        )
    else:
        console.print(
            "[yellow]⚠  No resume loaded. Use [bold]/load <path>[/bold] "
            "to upload a resume PDF.[/yellow]\n"
        )

    # ── Build the conversational RAG chain ───────────────────────────────────
    retriever = get_retriever()
    chain = build_conversational_chain(retriever)

    # ── Interactive loop ─────────────────────────────────────────────────────
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        if not user_input:
            continue

        # ── Command handling ─────────────────────────────────────────────────
        lower = user_input.lower()

        if lower in ("/exit", "/quit"):
            console.print("[dim]Goodbye! 👋[/dim]")
            break

        elif lower == "/help":
            print_help()

        elif lower == "/clear":
            clear_session(SESSION_ID)
            console.print("[green]✅ Conversation history cleared.[/green]\n")

        elif lower == "/history":
            messages = get_session_history_messages()
            if not messages:
                console.print("[dim]No conversation history yet.[/dim]\n")
            else:
                for msg in messages:
                    role = "You" if msg.type == "human" else "Assistant"
                    style = "blue" if msg.type == "human" else "green"
                    console.print(f"[{style}][{role}][/{style}]: {msg.content}\n")

        elif lower == "/links":
            print_links(current_links)

        elif lower == "/stats":
            print_stats()

        elif lower.startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print(
                    "[yellow]Usage: /load <path-to-resume.pdf>[/yellow]\n"
                )
                continue
            try:
                resolved = resolve_resume_path(parts[1].strip())
                meta = ingest_resume(resolved, reset=True)
                current_links = meta.links
                # Rebuild chain with fresh retriever after new data
                retriever = get_retriever()
                chain = build_conversational_chain(retriever)
                clear_session(SESSION_ID)
                console.print("[green]✅ New resume loaded. Chat history cleared.[/green]\n")
            except (FileNotFoundError, ValueError) as e:
                console.print(f"[red]❌ {e}[/red]\n")

        elif lower.startswith("/"):
            console.print(
                f"[yellow]Unknown command: {user_input}. "
                f"Type /help for available commands.[/yellow]\n"
            )

        # ── RAG Q&A ──────────────────────────────────────────────────────────
        else:
            if not collection_is_populated():
                console.print(
                    "[yellow]⚠  No resume loaded. "
                    "Use [bold]/load <path>[/bold] first.[/yellow]\n"
                )
                continue

            try:
                stream_and_print_answer(
                    lambda q=user_input: stream_conversational_answer(
                        chain, q, SESSION_ID
                    )
                )
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]\n")
