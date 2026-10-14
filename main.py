"""
AI Resume Bot — CLI entry point.

Usage:
    uv run main.py                        # Interactive mode (uses existing store)
    uv run main.py path/to/resume.pdf     # Load a resume and start chatting
    uv run main.py --help                 # Show usage
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-resume-bot",
        description="Interactive Context-Aware AI Resume Bot (RAG + Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py
  uv run main.py data/resumes/john_doe.pdf
  uv run main.py /absolute/path/to/resume.pdf
        """,
    )
    parser.add_argument(
        "resume",
        nargs="?",
        metavar="RESUME_PDF",
        help="Path to the resume PDF to load at startup (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate file extension early if provided
    if args.resume and not args.resume.lower().endswith(".pdf"):
        print(f"Error: '{args.resume}' is not a PDF file.", file=sys.stderr)
        sys.exit(1)

    # Import here to keep startup fast (lazy load heavy deps)
    from resume_bot.cli.interface import run_cli

    run_cli(pdf_path=args.resume)


if __name__ == "__main__":
    main()
