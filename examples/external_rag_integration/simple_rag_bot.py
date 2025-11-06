"""Minimal RAG chatbot demonstrating ACE integration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from rich import print as rich_print
from rich.panel import Panel

from agentic_context_engineering import Playbook
from agentic_context_engineering.agents import Generator
from agentic_context_engineering.utils.llm_client import LLMClient

# Avoid hard dependency during type checking/CI
if TYPE_CHECKING:  # pragma: no cover
    pass
typer: Any
try:  # runtime attempt
    import typer as _typer  # type: ignore

    typer = _typer
except Exception:  # pragma: no cover
    typer = cast(Any, object())

app = (
    typer.Typer(add_completion=False, help="Run a minimal RAG + ACE chatbot demo") if hasattr(typer, "Typer") else None
)


def load_playbook(playbook_path: Optional[str]) -> Playbook:
    if playbook_path and Path(playbook_path).exists():
        return Playbook.from_yaml(playbook_path)

    outputs_dir = Path("outputs")
    playbooks = sorted(outputs_dir.glob("playbook_v*.yaml"), key=lambda p: p.stat().st_mtime)
    if playbooks:
        return Playbook.from_yaml(str(playbooks[-1]))

    return Playbook.from_yaml("agentic_context_engineering/playbook_schema/base_playbook.yaml")


def toy_documents() -> List[Dict[str, Any]]:
    return [
        {
            "id": "course12_transformer_attn",
            "content": "Transformers use self-attention to weigh relationships between all tokens in a sequence.",
            "metadata": {"course": 12, "title": "Self-Attention Mechanics"},
        },
        {
            "id": "course13_finetuning",
            "content": "Fine-tuning requires preparing a dataset, selecting layers to update, and choosing optimisation parameters.",
            "metadata": {"course": 13, "title": "Fine-Tuning Workflow"},
        },
        {
            "id": "course4_python_basics",
            "content": "Python's list comprehensions provide a concise way to create lists from iterables.",
            "metadata": {"course": 4, "title": "Python Comprehensions"},
        },
    ]


def retrieve_docs(question: str, k: int = 2) -> List[Dict[str, Any]]:
    docs = toy_documents()
    question_tokens = set(question.lower().split())
    scored = []
    for doc in docs:
        doc_tokens = set(doc["content"].lower().split())
        overlap = len(question_tokens & doc_tokens)
        score = overlap / math.sqrt(len(doc_tokens) + 1)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored[:k]]


def pretty_print_docs(docs: Iterable[Dict[str, Any]]) -> None:
    for idx, doc in enumerate(docs, start=1):
        panel = Panel.fit(
            f"[bold]Document {idx}[/bold]\nID: {doc.get('id')}\nMetadata: {doc.get('metadata')}\nContent: {doc.get('content')}",
            title=f"[{idx}] Retrieved",
        )
        rich_print(panel)


@(app.command() if app else (lambda f: f))
def ask(
    question: str = typer.Argument(..., help="User question to answer"),
    playbook_path: Optional[str] = typer.Option(None, help="Path to playbook YAML"),
    offline: bool = typer.Option(False, help="Skip LLM call and show prompt only"),
):
    """Answer a question using ACE context plus toy retrieval results."""

    playbook = load_playbook(playbook_path)
    retrieved_docs = retrieve_docs(question)

    rich_print("[bold cyan]\nRetrieved Documents[/bold cyan]")
    pretty_print_docs(retrieved_docs)

    if offline:
        bundle = playbook.export_context_for_rag()
        prompt_lines = ["System Prompt:", bundle["system_prompt"], "", "Retrieved Context:"]
        for idx, doc in enumerate(retrieved_docs, start=1):
            prompt_lines.append(f"[{idx}] {doc.get('metadata', {}).get('title', doc.get('id'))}")
            prompt_lines.append(doc.get("content", ""))
            prompt_lines.append("")
        prompt_lines.append(f"Task: {question}")
        prompt_lines.append("Response:")
        rich_print(Panel("\n".join(prompt_lines), title="Prompt Preview"))
        if hasattr(typer, "Exit"):
            raise typer.Exit(code=0)
        return

    llm_client = LLMClient()
    generator = Generator(llm_client)
    response = generator.generate_with_context(question, playbook, retrieved_docs)
    citations = generator.extract_citations(response, retrieved_docs)

    rich_print("[bold green]\nAnswer[/bold green]")
    rich_print(response)

    rich_print("[bold magenta]\nCitations[/bold magenta]")
    for citation in citations:
        status = "✓" if citation["referenced"] else "?"
        rich_print(f"{status} {citation['label']} (confidence={citation['confidence']:.2f})")


if __name__ == "__main__" and app is not None:
    app()
