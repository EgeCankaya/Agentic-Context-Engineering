"""Integration-style tests for external ACE usage."""

from __future__ import annotations

from pathlib import Path

from agentic_context_engineering import ConversationLogger, Playbook
from agentic_context_engineering.agents import Generator


class FakeLLMClient:
    """Lightweight stand-in for LLMClient used in tests."""

    def generate(self, prompt: str, **_: object) -> str:  # pragma: no cover - trivial
        return (
            "Here is the answer you requested. It references the retrieved document [1] and follows the ACE heuristics."
        )


def sample_retrieved_docs():
    return [
        {
            "id": "course12_transformer_attn",
            "content": "Transformers rely on multi-head self-attention to capture context.",
            "metadata": {"course": 12, "title": "Self-Attention Mechanics"},
        }
    ]


def load_base_playbook() -> Playbook:
    return Playbook.from_yaml("agentic_context_engineering/playbook_schema/base_playbook.yaml")


def test_playbook_export_bundle_contains_required_keys():
    playbook = load_base_playbook()
    bundle = playbook.export_context_for_rag({"course": 12})

    assert "system_prompt" in bundle
    assert "heuristics" in bundle and isinstance(bundle["heuristics"], list)
    assert bundle["metadata"]["playbook_version"] == playbook.version


def test_generator_constructs_rag_prompt_with_retrieved_docs():
    playbook = load_base_playbook()
    generator = Generator(FakeLLMClient())
    docs = sample_retrieved_docs()
    prompt = generator.construct_rag_prompt("Explain self-attention", playbook, docs)

    assert "Retrieved Context" in prompt
    assert docs[0]["metadata"]["title"] in prompt


def test_generate_with_context_returns_citations():
    playbook = load_base_playbook()
    generator = Generator(FakeLLMClient())
    docs = sample_retrieved_docs()

    response = generator.generate_with_context("Explain", playbook, docs)
    citations = generator.extract_citations(response, docs)

    assert citations[0]["id"] == docs[0]["id"]
    assert citations[0]["referenced"] is True


def test_conversation_logger_exports_dataset(tmp_path: Path):
    logger = ConversationLogger(output_dir=tmp_path)
    session_id = logger.start_session(user_id="student_123", metadata={"course": 12})
    docs = sample_retrieved_docs()
    logger.log_turn(
        session_id=session_id,
        question="Explain self-attention",
        answer="Self-attention computes weights across tokens [1].",
        retrieved_docs=docs,
        annotations={"confidence": 0.9, "user_feedback": "helpful"},
    )
    logger.end_session(session_id)

    dataset = logger.export_for_ace()

    assert len(dataset) == 1
    entry = dataset[0]
    assert entry["input"] == "Explain self-attention"
    assert entry["metadata"]["turn_annotations"]["confidence"] == 0.9
