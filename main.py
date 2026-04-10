"""CLI entry point for the CodeOptimizer review workflow.

Usage examples::

    # Review a file
    python main.py --file my_script.py --language Python

    # Pass code directly (useful for short snippets)
    python main.py --code "def add(a, b): return a + b" --language Python

    # Use a different Ollama model
    python main.py --file main.py --model codellama

    # Point at a non-default Ollama instance
    python main.py --file main.py --ollama-url http://192.168.1.10:11434
"""

import argparse
import sys

from src.workflow import create_review_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeoptimizer",
        description="Agentic code review powered by LangGraph and a local Ollama model.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--file",
        metavar="PATH",
        help="Path to the source file to review.",
    )
    source.add_argument(
        "--code",
        metavar="CODE",
        help="Inline code snippet to review.",
    )
    parser.add_argument(
        "--language",
        metavar="LANG",
        default="",
        help="Programming language hint (e.g. Python, TypeScript).",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model tag (default: llama3.2).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Base URL of the Ollama server (default: http://localhost:11434).",
    )
    return parser


def _print_section(title: str, content: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}")
    print(title)
    print(separator)
    print(content)


def main(argv: list[str] | None = None) -> None:
    """Run the code review workflow from the command line."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            code = fh.read()
    elif args.code:
        code = args.code
    else:
        print("No --file or --code provided. Reading code from stdin (Ctrl-D to finish):")
        code = sys.stdin.read()

    workflow = create_review_workflow(model=args.model, base_url=args.ollama_url)

    initial_state = {
        "code": code,
        "language": args.language,
        "peer_review": "",
        "manager_review": "",
    }

    result = workflow.invoke(initial_state)

    _print_section("PEER REVIEW", result["peer_review"])
    _print_section("MANAGER REVIEW", result["manager_review"])


if __name__ == "__main__":
    main()
