"""
Pydantic models for ACE playbook schema.
Defines structured representation of evolving context for self-improving LLMs.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field


class Heuristic(BaseModel):
    """Individual heuristic rule in the playbook."""

    id: str
    rule: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    created_iteration: int = Field(ge=0, description="Iteration when created")
    last_updated: int = Field(ge=0, description="Last iteration updated")
    usage_count: int = Field(ge=0, default=0, description="Times used")
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0, description="Success rate 0-1")


class FewShotExample(BaseModel):
    """Few-shot example for the playbook."""

    input: str
    output: str
    # Make fields optional to satisfy tests that omit annotation and use explanation
    annotation: Optional[str] = None
    explanation: Optional[str] = None
    quality_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Quality score 0-1")


class PerformanceMetrics(BaseModel):
    """Performance metrics for a playbook version."""

    accuracy: float = Field(ge=0.0, le=1.0)
    bleu_score: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    rouge_score: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    exact_match: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    semantic_similarity: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    avg_tokens: int = Field(ge=0)
    inference_time_sec: float = Field(ge=0.0)
    vram_usage_gb: Optional[float] = Field(ge=0.0, default=None)


class HistoryEntry(BaseModel):
    """Entry in playbook change history."""

    iteration: int = Field(ge=0)
    change_type: Literal[
        "heuristic_added",
        "heuristic_updated",
        "heuristic_removed",
        "instruction_updated",
        "example_added",
        "example_updated",
        "example_removed",
    ]
    description: str
    performance_delta: Optional[float] = Field(default=None, description="Performance change")
    heuristic_id: Optional[str] = Field(default=None, description="Related heuristic ID")


class PlaybookContext(BaseModel):
    """Core context content of the playbook."""

    system_instructions: str
    heuristics: List[Heuristic] = Field(default_factory=list)
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class PlaybookMetadata(BaseModel):
    """Metadata for a playbook version."""

    created_at: Union[datetime, str]
    iteration: int = Field(ge=0)
    parent_version: Optional[str] = Field(default=None)
    performance_metrics: PerformanceMetrics
    convergence_status: Literal["improving", "plateaued", "degraded"]
    vram_usage_gb: Optional[float] = Field(ge=0.0, default=None)


class Playbook(BaseModel):
    """Complete playbook with version, metadata, context, and history."""

    version: str = Field(pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    metadata: PlaybookMetadata
    context: PlaybookContext
    history: List[HistoryEntry] = Field(default_factory=list)

    def to_yaml(self, path: str) -> None:
        """Save playbook to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Playbook":
        """Load playbook from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def add_heuristic(self, rule: str, confidence: float, iteration: int) -> str:
        """Add a new heuristic to the playbook."""
        heuristic_id = f"h{len(self.context.heuristics) + 1:03d}"
        heuristic = Heuristic(
            id=heuristic_id, rule=rule, confidence=confidence, created_iteration=iteration, last_updated=iteration
        )
        self.context.heuristics.append(heuristic)

        # Add to history
        self.history.append(
            HistoryEntry(
                iteration=iteration,
                change_type="heuristic_added",
                description=f"Added heuristic {heuristic_id}: {rule[:50]}...",
                heuristic_id=heuristic_id,
            )
        )

        return heuristic_id

    def update_heuristic(self, heuristic_id: str, rule: str, confidence: float, iteration: int) -> bool:
        """Update an existing heuristic."""
        for heuristic in self.context.heuristics:
            if heuristic.id == heuristic_id:
                heuristic.rule = rule
                heuristic.confidence = confidence
                heuristic.last_updated = iteration

                # Add to history
                self.history.append(
                    HistoryEntry(
                        iteration=iteration,
                        change_type="heuristic_updated",
                        description=f"Updated heuristic {heuristic_id}",
                        heuristic_id=heuristic_id,
                    )
                )
                return True
        return False

    def add_example(
        self, input_text: str, output_text: str, annotation: str, quality_score: float, iteration: int
    ) -> None:
        """Add a few-shot example to the playbook."""
        example = FewShotExample(
            input=input_text, output=output_text, annotation=annotation, quality_score=quality_score
        )
        self.context.few_shot_examples.append(example)

        # Add to history
        self.history.append(
            HistoryEntry(
                iteration=iteration, change_type="example_added", description=f"Added example: {input_text[:50]}..."
            )
        )

    def get_system_prompt(
        self,
        include_guidelines: bool = True,
        include_constraints: bool = True,
    ) -> str:
        """Assemble a complete system prompt from the playbook contents."""

        sections: List[str] = []
        instructions = (self.context.system_instructions or "").strip()
        if instructions:
            sections.append(instructions)

        if include_guidelines and self.context.heuristics:
            sections.append("Guidelines:")
            for heuristic in self.context.heuristics:
                sections.append(f"- {heuristic.rule}")

        if include_constraints and self.context.constraints:
            sections.append("Constraints:")
            for constraint in self.context.constraints:
                sections.append(f"- {constraint}")

        return "\n".join(sections).strip()

    def export_context_for_rag(
        self,
        task_metadata: Optional[Dict[str, Any]] = None,
        top_k_examples: int = 3,
    ) -> Dict[str, Any]:
        """Export playbook context as a structured bundle for RAG systems."""

        relevant_examples = self.get_relevant_examples(task_metadata, top_k=top_k_examples)

        return {
            "system_prompt": self.get_system_prompt(),
            "heuristics": [heuristic.model_dump() for heuristic in self.context.heuristics],
            "examples": [example.model_dump() for example in relevant_examples],
            "constraints": list(self.context.constraints),
            "metadata": {
                "playbook_version": self.version,
                "iteration": self.metadata.iteration,
                "playbook_metadata": self.metadata.model_dump(),
                "task_metadata": task_metadata or {},
            },
        }

    def get_relevant_examples(
        self,
        task: Optional[Union[str, Dict[str, Any]]] = None,
        top_k: int = 3,
    ) -> List[FewShotExample]:
        """Return few-shot examples that best match a given task description."""

        if not self.context.few_shot_examples:
            return []

        if task is None:
            # Return highest quality examples by default
            return sorted(
                self.context.few_shot_examples,
                key=lambda ex: ex.quality_score,
                reverse=True,
            )[:top_k]

        task_text = self._normalise_task(task)

        scored_examples: List[Tuple[float, FewShotExample]] = []
        for example in self.context.few_shot_examples:
            score = self._score_example(example, task_text)
            scored_examples.append((score, example))

        scored_examples.sort(key=lambda item: item[0], reverse=True)
        return [example for score, example in scored_examples[:top_k] if score > 0]

    def format_for_llm(
        self,
        task_type: Optional[str] = None,
        include_examples: bool = True,
    ) -> str:
        """Create a formatted prompt template ready for LLM consumption."""

        lines: List[str] = [self.get_system_prompt()]

        if task_type:
            lines.append("")
            lines.append(f"Task Type: {task_type}")

        if include_examples:
            examples = self.get_relevant_examples(task_type, top_k=3)
            if examples:
                lines.append("")
                lines.append("Relevant Examples:")
                for example in examples:
                    lines.append(f"- Input: {example.input}")
                    lines.append(f"  Output: {example.output}")
                    if example.annotation:
                        lines.append(f"  Note: {example.annotation}")

        lines.append("")
        lines.append("Task: {task}")
        lines.append("Response:")
        return "\n".join(lines)

    def get_heuristic_by_id(self, heuristic_id: str) -> Optional[Heuristic]:
        """Get heuristic by ID."""
        for heuristic in self.context.heuristics:
            if heuristic.id == heuristic_id:
                return heuristic
        return None

    def get_heuristics_by_iteration(self, iteration: int) -> List[Heuristic]:
        """Get heuristics created in a specific iteration."""
        return [h for h in self.context.heuristics if h.created_iteration == iteration]

    def get_recent_heuristics(self, n: int = 5) -> List[Heuristic]:
        """Get the N most recently created heuristics."""
        return sorted(self.context.heuristics, key=lambda h: h.created_iteration, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _normalise_task(self, task: Optional[Union[str, Dict[str, Any]]]) -> str:
        if isinstance(task, dict):
            return " ".join(str(value) for value in task.values()).lower()
        if isinstance(task, str):
            return task.lower()
        return ""

    def _score_example(self, example: FewShotExample, task_text: str) -> float:
        if not task_text:
            return example.quality_score

        match_score = 0.0
        example_text = f"{example.input} {example.output} {example.annotation}".lower()

        for token in task_text.split():
            if token and token in example_text:
                match_score += 1.0

        if match_score == 0:
            return example.quality_score * 0.5

        return match_score + example.quality_score
