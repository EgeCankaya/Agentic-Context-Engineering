"""
Generator Agent for ACE framework.
Produces model outputs based on current playbook context.
"""

import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

from ..playbook_schema import Playbook
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class Generator:
    """
    Generator agent that produces outputs using the current playbook context.

    The generator constructs prompts from the playbook's system instructions,
    heuristics, and few-shot examples, then uses the LLM to generate responses.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize Generator agent.

        Args:
            llm_client: LLM client for text generation
        """
        self.llm_client = llm_client
        logger.info("Generator agent initialized")

    def generate(self, task: str, playbook: Playbook, **kwargs) -> str:
        """
        Generate response for a task using the current playbook.

        Args:
            task: Input task/question
            playbook: Current playbook with context
            **kwargs: Additional parameters for LLM generation

        Returns:
            Generated response
        """
        # Construct prompt from playbook context
        prompt = self.construct_prompt(task, playbook)

        # Generate response using LLM
        response = self.llm_client.generate(prompt, **kwargs)

        logger.debug(f"Generated response for task: {task[:50]}...")
        return response

    def generate_batch(self, tasks: list[str], playbook: Playbook, **kwargs) -> list[str]:
        """
        Generate responses for multiple tasks.

        Args:
            tasks: List of input tasks
            playbook: Current playbook with context
            **kwargs: Additional parameters for LLM generation

        Returns:
            List of generated responses
        """
        responses = []
        for task in tasks:
            response = self.generate(task, playbook, **kwargs)
            responses.append(response)
        return responses

    def construct_prompt(self, task: str, playbook: Playbook) -> str:
        """
        Construct prompt from playbook context.

        Args:
            task: Input task
            playbook: Current playbook

        Returns:
            Constructed prompt
        """
        prompt_parts = []

        # System instructions
        if playbook.context.system_instructions:
            prompt_parts.append(f"System Instructions:\n{playbook.context.system_instructions}\n")

        # Heuristics
        if playbook.context.heuristics:
            prompt_parts.append("Guidelines:")
            for heuristic in playbook.context.heuristics:
                prompt_parts.append(f"- {heuristic.rule}")
            prompt_parts.append("")

        # Few-shot examples
        if playbook.context.few_shot_examples:
            prompt_parts.append("Examples:")
            for example in playbook.context.few_shot_examples:
                prompt_parts.append(f"Input: {example.input}")
                prompt_parts.append(f"Output: {example.output}")
                if example.annotation:
                    prompt_parts.append(f"Note: {example.annotation}")
                prompt_parts.append("")

        # Constraints
        if playbook.context.constraints:
            prompt_parts.append("Constraints:")
            for constraint in playbook.context.constraints:
                prompt_parts.append(f"- {constraint}")
            prompt_parts.append("")

        # Task
        prompt_parts.append(f"Task: {task}")
        prompt_parts.append("Response:")

        return "\n".join(prompt_parts)

    # Backwards compatibility with previous private method name
    def _construct_prompt(self, task: str, playbook: Playbook) -> str:  # pragma: no cover - legacy shim
        return self.construct_prompt(task, playbook)

    def construct_rag_prompt(
        self,
        task: str,
        playbook: Playbook,
        retrieved_docs: Iterable[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct a prompt that weaves retrieved knowledge into the playbook context."""

        template = self.construct_prompt("{task}", playbook)
        retrieved_section = self._format_retrieved_docs(retrieved_docs, metadata)

        return template.replace(
            "Task: {task}",
            f"Retrieved Context:\n{retrieved_section}\nTask: {task}",
        )

    def generate_with_context(
        self,
        task: str,
        playbook: Playbook,
        retrieved_docs: Iterable[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate an answer using retrieved documents alongside ACE heuristics."""

        prompt = self.construct_rag_prompt(task, playbook, retrieved_docs, metadata=metadata)
        response = self.llm_client.generate(prompt, **kwargs)
        logger.debug("Generated RAG-enhanced response for task: %s", task[:50])
        return response

    def extract_citations(
        self,
        response: str,
        retrieved_docs: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify which retrieved documents are referenced in a model response."""

        citations: List[Dict[str, Any]] = []
        for index, doc in enumerate(retrieved_docs, start=1):
            doc_id = str(doc.get("id") or f"doc_{index}")
            metadata = doc.get("metadata", {})
            citation_label = metadata.get("title") or metadata.get("heading") or doc_id
            indicator = f"[{index}]"

            matched = False
            text = response.lower()
            if indicator.lower() in text or doc_id.lower() in text:
                matched = True
            else:
                title = str(citation_label).lower()
                if title and title in text:
                    matched = True
                else:
                    snippet = str(doc.get("content", "")).lower()[:120]
                    if snippet and snippet[:40] in text:
                        matched = True

            citations.append({
                "id": doc_id,
                "label": citation_label,
                "referenced": matched,
                "metadata": metadata,
                "confidence": 1.0 if matched else 0.25,
            })

        return citations

    def get_prompt_template(self, playbook: Playbook) -> str:
        """
        Get the prompt template used by this generator.

        Args:
            playbook: Current playbook

        Returns:
            Prompt template string
        """
        return self.construct_prompt("{TASK_PLACEHOLDER}", playbook).replace("Task: {TASK_PLACEHOLDER}", "Task: {task}")

    def analyze_prompt_usage(self, playbook: Playbook) -> Dict[str, Any]:
        """
        Analyze how the playbook context is being used in prompts.

        Args:
            playbook: Current playbook

        Returns:
            Analysis of prompt usage
        """
        analysis = {
            "system_instructions_length": len(playbook.context.system_instructions),
            "heuristics_count": len(playbook.context.heuristics),
            "examples_count": len(playbook.context.few_shot_examples),
            "constraints_count": len(playbook.context.constraints),
            "total_context_length": 0,
        }

        # Calculate total context length
        context_parts = [
            playbook.context.system_instructions,
            *[h.rule for h in playbook.context.heuristics],
            *[f"{ex.input} {ex.output}" for ex in playbook.context.few_shot_examples],
            *playbook.context.constraints,
        ]

        analysis["total_context_length"] = sum(len(part) for part in context_parts)

        # Analyze heuristic usage
        analysis["heuristic_usage"] = {
            "high_usage": [h for h in playbook.context.heuristics if h.usage_count > 10],
            "high_success": [h for h in playbook.context.heuristics if h.success_rate > 0.8],
            "recent": [
                h for h in playbook.context.heuristics if h.created_iteration >= playbook.metadata.iteration - 2
            ],
        }

        return analysis

    def optimize_prompt(self, playbook: Playbook, max_length: int = 4000) -> Playbook:
        """
        Optimize playbook context for prompt length.

        Args:
            playbook: Current playbook
            max_length: Maximum prompt length

        Returns:
            Optimized playbook
        """
        # Create a copy to avoid modifying original
        optimized_playbook = playbook.model_copy(deep=True)

        # Remove low-usage heuristics if context is too long
        current_length = self._estimate_prompt_length(optimized_playbook)

        if current_length > max_length:
            # Sort heuristics by usage and success rate
            heuristics = sorted(
                optimized_playbook.context.heuristics, key=lambda h: (h.usage_count, h.success_rate), reverse=True
            )

            # Keep only top heuristics
            optimized_playbook.context.heuristics = heuristics[:5]

            # Remove oldest examples if still too long
            if self._estimate_prompt_length(optimized_playbook) > max_length:
                optimized_playbook.context.few_shot_examples = optimized_playbook.context.few_shot_examples[-3:]

        return optimized_playbook

    def _estimate_prompt_length(self, playbook: Playbook) -> int:
        """Estimate the length of a prompt constructed from the playbook."""
        template = self.construct_prompt("Sample task", playbook)
        return len(template)

    def validate_response(self, response: str, task: str) -> Dict[str, Any]:
        """
        Validate generated response against basic criteria.

        Args:
            response: Generated response
            task: Original task

        Returns:
            Validation results
        """
        validation = {
            "length_ok": 50 <= len(response) <= 2000,
            "not_empty": len(response.strip()) > 0,
            "contains_task_keywords": any(word in response.lower() for word in task.lower().split()),
            "has_structure": any(marker in response for marker in ["```", "##", "-", "1.", "2."]),
            "issues": [],
        }

        # Check for issues
        if not validation["length_ok"]:
            validation["issues"].append("Response length outside acceptable range")

        if not validation["not_empty"]:
            validation["issues"].append("Response is empty")

        if not validation["contains_task_keywords"]:
            validation["issues"].append("Response doesn't address task keywords")

        if not validation["has_structure"]:
            validation["issues"].append("Response lacks structure (no code, headers, lists)")

        return validation

    def _format_retrieved_docs(
        self,
        retrieved_docs: Iterable[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Format retrieved documents for inclusion in the RAG prompt."""

        lines: List[str] = []
        if metadata:
            lines.append("Query Metadata:")
            for key, value in metadata.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        for index, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc.get("id") or f"doc_{index}"
            title = doc.get("metadata", {}).get("title")
            heading = doc.get("metadata", {}).get("heading")
            label = title or heading or doc_id
            content = doc.get("content") or doc.get("text") or ""
            content = content.strip()

            lines.append(f"[{index}] {label}")
            if doc.get("metadata"):
                lines.append(f"  Metadata: {doc['metadata']}")
            if content:
                preview = content if len(content) <= 500 else content[:500] + "..."
                lines.append(f"  Content: {preview}")
            lines.append("")

        if not lines:
            lines.append("No retrieved documents were available.")

        return "\n".join(lines)
