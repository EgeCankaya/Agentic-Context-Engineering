"""Utility for logging multi-turn conversations for ACE feedback."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConversationTurn:
    """Single turn within a conversation session."""

    question: str
    answer: str
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConversationSession:
    """Metadata and content for a conversation session."""

    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    closed_at: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the session to a JSON-serialisable dictionary."""

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "turns": [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "retrieved_docs": turn.retrieved_docs,
                    "annotations": turn.annotations,
                    "timestamp": turn.timestamp,
                }
                for turn in self.turns
            ],
        }


class ConversationLogger:
    """Helper class for logging conversations and exporting ACE datasets."""

    def __init__(
        self,
        output_dir: str = "outputs/conversations",
        auto_save: bool = True,
        max_turns_per_session: Optional[int] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.max_turns_per_session = max_turns_per_session
        self._sessions: Dict[str, ConversationSession] = {}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def start_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Start a new conversation session."""

        session_id = session_id or uuid.uuid4().hex
        if session_id in self._sessions:
            raise ValueError(f"Session '{session_id}' already exists")

        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        return session_id

    def end_session(self, session_id: str) -> None:
        """Close a session and persist it to disk if auto-save is enabled."""

        session = self._require_session(session_id)
        session.closed_at = datetime.now(timezone.utc).isoformat()
        if self.auto_save:
            self._save_session(session)

    def get_session_history(self, session_id: str) -> ConversationSession:
        """Return the conversation session object."""

        return self._require_session(session_id)

    # ------------------------------------------------------------------
    # Logging and feedback
    # ------------------------------------------------------------------
    def log_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        retrieved_docs: Optional[Iterable[Dict[str, Any]]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single Q&A turn."""

        session = self._require_session(session_id)

        if self.max_turns_per_session and len(session.turns) >= self.max_turns_per_session:
            raise ValueError(f"Session '{session_id}' exceeded max turns ({self.max_turns_per_session})")

        turn = ConversationTurn(
            question=question,
            answer=answer,
            retrieved_docs=list(retrieved_docs or []),
            annotations=annotations or {},
        )
        session.turns.append(turn)

        if self.auto_save:
            self._save_session(session)

    def append_feedback(
        self,
        session_id: str,
        turn_index: int,
        feedback: Dict[str, Any],
    ) -> None:
        """Attach extra feedback/annotations to a specific turn."""

        session = self._require_session(session_id)
        try:
            session.turns[turn_index].annotations.update(feedback)
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise IndexError(f"Turn {turn_index} not found in session '{session_id}'") from exc

        if self.auto_save:
            self._save_session(session)

    # ------------------------------------------------------------------
    # Export utilities
    # ------------------------------------------------------------------
    def export_for_ace(self) -> List[Dict[str, Any]]:
        """Export all logged conversations as ACE-compatible dataset entries."""

        dataset: List[Dict[str, Any]] = []
        for session in self._sessions.values():
            for turn_index, turn in enumerate(session.turns):
                dataset.append(self._turn_to_ace_item(session, turn, turn_index))
        return dataset

    def calculate_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Derive lightweight metrics (e.g., average confidence) for a session."""

        session = self._require_session(session_id)
        confidences: List[float] = []
        helpful_votes = 0
        for turn in session.turns:
            confidence = turn.annotations.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
            if str(turn.annotations.get("user_feedback", "")).lower() in {"helpful", "positive"}:
                helpful_votes += 1

        avg_conf = sum(confidences) / len(confidences) if confidences else None
        return {
            "session_id": session_id,
            "num_turns": len(session.turns),
            "average_confidence": avg_conf,
            "helpful_votes": helpful_votes,
        }

    def flush(self) -> None:
        """Persist all sessions to disk regardless of auto-save setting."""

        for session in self._sessions.values():
            self._save_session(session)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_session(self, session_id: str) -> ConversationSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Session '{session_id}' not found") from exc

    def _turn_to_ace_item(
        self,
        session: ConversationSession,
        turn: ConversationTurn,
        turn_index: int,
    ) -> Dict[str, Any]:
        """Convert a conversation turn into ACE dataset format."""

        retrieved_meta = [
            {
                "id": doc.get("id"),
                "score": doc.get("score"),
                "metadata": doc.get("metadata", {}),
            }
            for doc in turn.retrieved_docs
        ]

        evaluation_criteria = turn.annotations.get("evaluation_criteria") or {
            "accuracy": "Must be factually correct and grounded in retrieved documents.",
            "grounding": "Cite relevant course/module references when available.",
            "completeness": "Cover all sub-questions raised by the learner.",
        }

        metadata = {
            "session_id": session.session_id,
            "turn_index": turn_index,
            "user_id": session.user_id,
            "session_metadata": session.metadata,
            "turn_annotations": turn.annotations,
            "retrieved_docs": retrieved_meta,
            "timestamp": turn.timestamp,
        }

        return {
            "id": f"{session.session_id}_turn_{turn_index}",
            "input": turn.question,
            "reference_output": turn.answer,
            "evaluation_criteria": evaluation_criteria,
            "metadata": metadata,
        }

    def _save_session(self, session: ConversationSession) -> None:
        """Persist a session as JSON in the output directory."""

        output_path = self.output_dir / f"{session.session_id}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(session.to_dict(), handle, indent=2, ensure_ascii=False)
