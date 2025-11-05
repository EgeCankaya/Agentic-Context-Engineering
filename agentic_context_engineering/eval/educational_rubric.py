"""Domain-specific evaluator for educational conversations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional


@dataclass
class MetricResult:
    """Container for a single metric result."""

    name: str
    score: float
    rationale: str


class EducationalEvaluator:
    """Evaluate chatbot answers against educational quality criteria."""

    def __init__(self) -> None:
        self.pedagogy_markers = [
            "step-by-step",
            "let's break",
            "first",
            "next",
            "in summary",
            "remember",
        ]
        self.clarity_max_sentence_length = 30

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_conversation(
        self,
        question: str,
        answer: str,
        retrieved_docs: Optional[Iterable[Dict[str, Any]]] = None,
        expectations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score the given answer using education-focused metrics."""

        retrieved_docs = list(retrieved_docs or [])
        expectations = expectations or {}

        metrics: List[MetricResult] = [
            self._score_accuracy(answer, expectations),
            self._score_grounding(answer, retrieved_docs, expectations),
            self._score_citation_quality(answer, retrieved_docs),
            self._score_completeness(answer, expectations),
            self._score_pedagogy(answer),
            self._score_clarity(answer),
            self._score_idk(question, answer, expectations),
        ]

        scores = {metric.name: metric.score for metric in metrics}
        rationales = {metric.name: metric.rationale for metric in metrics}

        overall = mean(scores.values()) if scores else 0.0

        return {
            "scores": scores,
            "rationales": rationales,
            "overall": overall,
            "recommendations": self._make_recommendations(scores, rationales),
        }

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------
    def _score_accuracy(self, answer: str, expectations: Dict[str, Any]) -> MetricResult:
        must_include = expectations.get("must_include", [])
        should_include = expectations.get("should_include", [])

        score = 1.0
        missing_must = [term for term in must_include if term.lower() not in answer.lower()]
        missing_should = [term for term in should_include if term.lower() not in answer.lower()]

        if missing_must:
            score -= 0.6
        if missing_should:
            score -= 0.2

        score = max(0.0, min(score, 1.0))
        rationale = (
            "Covers required concepts" if score > 0.7 else "Missing required concepts: " + ", ".join(missing_must)
        )
        return MetricResult("accuracy", score, rationale)

    def _score_grounding(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        expectations: Dict[str, Any],
    ) -> MetricResult:
        if not retrieved_docs:
            return MetricResult("grounding", 0.5, "No retrieved documents provided")

        citations = re.findall(r"\[(\d+)\]", answer)
        expected_sources = expectations.get("preferred_sources") or []
        referenced_indexes = {int(cite) for cite in citations if cite.isdigit()}

        matched_sources = 0
        for idx, doc in enumerate(retrieved_docs, start=1):
            if (
                idx in referenced_indexes
                or (doc.get("id") and doc["id"] in answer)
                or (doc.get("metadata", {}).get("title") and doc["metadata"]["title"] in answer)
            ):
                matched_sources += 1

        bonus = 0.1 if expected_sources and any(src in answer for src in expected_sources) else 0.0

        score = (matched_sources / len(retrieved_docs)) + bonus
        score = max(0.0, min(score, 1.0))
        rationale = f"Referenced {matched_sources}/{len(retrieved_docs)} documents"
        return MetricResult("grounding", score, rationale)

    def _score_citation_quality(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> MetricResult:
        if not retrieved_docs:
            return MetricResult("citation_quality", 0.5, "No retrieved documents provided")

        citations = re.findall(r"\[(\d+)\]", answer)
        if not citations:
            return MetricResult("citation_quality", 0.2, "Response lacks explicit citations")

        unique_citations = {int(c) for c in citations if c.isdigit()}
        coverage = len(unique_citations) / len(retrieved_docs)
        score = max(0.2, min(1.0, coverage + 0.2))
        rationale = f"Uses {len(unique_citations)} distinct citations"
        return MetricResult("citation_quality", score, rationale)

    def _score_completeness(self, answer: str, expectations: Dict[str, Any]) -> MetricResult:
        subtopics = expectations.get("subtopics") or []
        if not subtopics:
            return MetricResult("completeness", 0.8, "No subtopics specified")

        covered = [topic for topic in subtopics if topic.lower() in answer.lower()]
        score = len(covered) / len(subtopics)
        rationale = f"Covered {len(covered)}/{len(subtopics)} subtopics"
        return MetricResult("completeness", score, rationale)

    def _score_pedagogy(self, answer: str) -> MetricResult:
        lower = answer.lower()
        markers = sum(1 for marker in self.pedagogy_markers if marker in lower)
        structured = any(token in answer for token in ["1.", "2.", "- ", "\n\n"])

        score = min(1.0, 0.3 + markers * 0.15 + (0.2 if structured else 0.0))
        rationale = "Provides learner-friendly structure" if score > 0.6 else "Could use clearer instructional framing"
        return MetricResult("pedagogy", score, rationale)

    def _score_clarity(self, answer: str) -> MetricResult:
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
        if not sentences:
            return MetricResult("clarity", 0.0, "No sentences detected")

        long_sentences = [s for s in sentences if len(s.split()) > self.clarity_max_sentence_length]
        ratio = len(long_sentences) / len(sentences)
        score = max(0.1, 1.0 - ratio)
        rationale = "Concise explanations" if score > 0.7 else "Contains long or complex sentences"
        return MetricResult("clarity", score, rationale)

    def _score_idk(self, question: str, answer: str, expectations: Dict[str, Any]) -> MetricResult:
        should_deflect = expectations.get("should_deflect", False)
        mentions_idk = "i don't know" in answer.lower()

        if should_deflect:
            score = 1.0 if mentions_idk else 0.1
            rationale = "Properly acknowledged uncertainty" if score > 0.5 else "Should have deflected"
        else:
            score = 0.9 if not mentions_idk else 0.3
            rationale = "Provided a confident answer" if score > 0.5 else "Unnecessary deflection"

        return MetricResult("idk_handling", score, rationale)

    # ------------------------------------------------------------------
    # Recommendation helper
    # ------------------------------------------------------------------
    def _make_recommendations(
        self,
        scores: Dict[str, float],
        rationales: Dict[str, str],
    ) -> List[str]:
        recommendations: List[str] = []
        for name, score in scores.items():
            if score < 0.5:
                recommendations.append(f"Improve {name}: {rationales.get(name, '')}")
        return recommendations
