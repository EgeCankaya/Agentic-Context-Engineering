"""Log a demo conversation using the ConversationLogger utility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

from rich import print as rich_print

from agentic_context_engineering import ConversationLogger

# Avoid hard dependency during type checking/CI
if TYPE_CHECKING:  # pragma: no cover
    pass
typer: Any
try:  # runtime attempt
    import typer as _typer  # type: ignore

    typer = _typer
except Exception:  # pragma: no cover - allow running without typer installed
    typer = cast(Any, object())

try:  # Support running as script or module
    from .simple_rag_bot import load_playbook, retrieve_docs  # type: ignore
except ImportError:  # pragma: no cover - executed when run as standalone script
    import sys

    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent.parent))
    from examples.external_rag_integration.simple_rag_bot import load_playbook, retrieve_docs  # type: ignore


app = typer.Typer(add_completion=False) if hasattr(typer, "Typer") else None


@(app.command() if app else (lambda f: f))
def log_demo(
    question: str = typer.Option(
        "Explain how transformers use self-attention",
        help="Question to log",
    ),
    answer_path: Optional[str] = typer.Option(
        None,
        help="Path to a text file containing the assistant answer",
    ),
    output_dataset: str = typer.Option(
        "outputs/conversations/sample_dataset.json",
        help="Path to save ACE-formatted dataset entries",
    ),
):
    """Create a simple conversation log ready for ACE ingestion."""

    playbook = load_playbook(None)
    retrieved_docs = retrieve_docs(question)

    if answer_path:
        answer = Path(answer_path).read_text(encoding="utf-8")
    else:
        answer = (
            "Transformers rely on self-attention to compare every token with each other. "
            "Each head focuses on different relationships, and the results are concatenated before "
            "passing through feed-forward layers."
        )

    logger = ConversationLogger(output_dir="outputs/conversations")
    session_id = logger.start_session(user_id="demo_student", metadata={"playbook_version": playbook.version})
    logger.log_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        retrieved_docs=retrieved_docs,
        annotations={
            "confidence": 0.75,
            "user_feedback": "helpful",
            "evaluation_criteria": {
                "accuracy": "Must mention self-attention mechanics",
                "grounding": "Reference Course 12",
                "pedagogy": "Explain in learner-friendly terms",
            },
        },
    )
    logger.end_session(session_id)

    dataset = logger.export_for_ace()
    output_path = Path(output_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    rich_print(f"Saved conversation session to [bold]{output_path}[/bold]")
    rich_print(dataset)


if __name__ == "__main__" and app is not None:
    app()
