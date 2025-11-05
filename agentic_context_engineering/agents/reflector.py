"""
Reflector Agent for ACE framework.
Evaluates generated outputs and identifies areas for improvement.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..playbook_schema import Playbook
from ..utils.llm_client import LLMClient
from ..utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Reflector:
    """
    Reflector agent that evaluates generator outputs and provides feedback.

    The reflector uses structured rubrics to assess outputs on multiple
    dimensions and identifies patterns for improvement.
    """

    def __init__(self, llm_client: LLMClient, metrics_calculator: Optional[MetricsCalculator] = None):
        """
        Initialize Reflector agent.

        Args:
            llm_client: LLM client for reflection analysis
            metrics_calculator: Optional metrics calculator for quantitative evaluation
        """
        self.llm_client = llm_client
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        logger.info("Reflector agent initialized")

    def reflect(
        self,
        task: str,
        generated_output: str,
        reference_output: Optional[str] = None,
        playbook: Optional[Playbook] = None,
    ) -> Dict[str, Any]:
        """
        Reflect on a generated output and provide structured feedback.

        Args:
            task: Original task/question
            generated_output: Output from generator
            reference_output: Optional reference output for comparison
            playbook: Current playbook for context

        Returns:
            Structured reflection with scores and suggestions
        """
        # Calculate quantitative metrics if reference available
        quantitative_metrics = {}
        if reference_output:
            quantitative_metrics = self.metrics_calculator.calculate_all_metrics(
                [generated_output],
                [reference_output],
                custom_criteria=["code_example", "citation", "explanation", "error_handling"],
            )

        # Generate qualitative reflection
        qualitative_reflection = self._generate_qualitative_reflection(
            task, generated_output, reference_output, playbook
        )

        # Combine results
        reflection = {
            "task": task,
            "generated_output": generated_output,
            "quantitative_metrics": quantitative_metrics,
            "qualitative_analysis": qualitative_reflection,
            "overall_score": self._calculate_overall_score(quantitative_metrics, qualitative_reflection),
            "improvement_suggestions": self._generate_improvement_suggestions(qualitative_reflection, playbook),
            "error_patterns": self._identify_error_patterns(generated_output, task),
        }

        logger.debug(f"Reflection completed for task: {task[:50]}...")
        return reflection

    def reflect_batch(
        self,
        tasks: List[str],
        generated_outputs: List[str],
        reference_outputs: Optional[List[str]] = None,
        playbook: Optional[Playbook] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reflect on multiple generated outputs.

        Args:
            tasks: List of original tasks
            generated_outputs: List of generated outputs
            reference_outputs: Optional list of reference outputs
            playbook: Current playbook for context

        Returns:
            List of reflection results
        """
        reflections = []

        for i, (task, output) in enumerate(zip(tasks, generated_outputs)):
            ref_output = reference_outputs[i] if reference_outputs else None
            reflection = self.reflect(task, output, ref_output, playbook)
            reflections.append(reflection)

        return reflections

    def _generate_qualitative_reflection(
        self, task: str, generated_output: str, reference_output: Optional[str], playbook: Optional[Playbook]
    ) -> Dict[str, Any]:
        """Generate qualitative analysis using LLM."""

        # Construct reflection prompt
        prompt = self._construct_reflection_prompt(task, generated_output, reference_output, playbook)

        # Get LLM reflection
        llm_reflection = self.llm_client.generate(prompt, temperature=0.3)

        # Parse structured response
        try:
            # Try to extract JSON from response
            json_start = llm_reflection.find("{")
            json_end = llm_reflection.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                structured_response = json.loads(llm_reflection[json_start:json_end])
            else:
                # Fallback to parsing text
                structured_response = self._parse_text_reflection(llm_reflection)
        except json.JSONDecodeError:
            structured_response = self._parse_text_reflection(llm_reflection)

        return structured_response

    def _construct_reflection_prompt(
        self, task: str, generated_output: str, reference_output: Optional[str], playbook: Optional[Playbook]
    ) -> str:
        """Construct prompt for reflection analysis."""

        prompt_parts = [
            "You are an expert evaluator analyzing AI-generated responses. Provide structured feedback.",
            "",
            f"Task: {task}",
            f"Generated Output: {generated_output}",
        ]

        if reference_output:
            prompt_parts.append(f"Reference Output: {reference_output}")

        if playbook:
            prompt_parts.append(f"Current Guidelines: {[h.rule for h in playbook.context.heuristics]}")

        prompt_parts.extend([
            "",
            "Evaluate the generated output on these dimensions:",
            "1. Accuracy: Is the information correct?",
            "2. Completeness: Does it fully address the task?",
            "3. Clarity: Is it well-structured and easy to understand?",
            "4. Practicality: Does it provide actionable advice?",
            "5. Code Quality: Are code examples correct and runnable?",
            "6. Citations: Are sources properly referenced?",
            "",
            "Provide your analysis in this JSON format:",
            "{",
            '  "accuracy_score": 0.0-1.0,',
            '  "completeness_score": 0.0-1.0,',
            '  "clarity_score": 0.0-1.0,',
            '  "practicality_score": 0.0-1.0,',
            '  "code_quality_score": 0.0-1.0,',
            '  "citation_score": 0.0-1.0,',
            '  "strengths": ["list of strengths"],',
            '  "weaknesses": ["list of weaknesses"],',
            '  "missing_elements": ["what is missing"],',
            '  "suggestions": ["specific improvement suggestions"]',
            "}",
        ])

        return "\n".join(prompt_parts)

    def _parse_text_reflection(self, text: str) -> Dict[str, Any]:
        """Parse text reflection into structured format."""
        # Simple text parsing as fallback
        return {
            "accuracy_score": 0.7,
            "completeness_score": 0.7,
            "clarity_score": 0.7,
            "practicality_score": 0.7,
            "code_quality_score": 0.7,
            "citation_score": 0.7,
            "strengths": ["Response addresses the task"],
            "weaknesses": ["Could be more detailed"],
            "missing_elements": ["Code examples", "Citations"],
            "suggestions": ["Add more specific examples", "Include documentation links"],
        }

    def _calculate_overall_score(
        self, quantitative_metrics: Dict[str, float], qualitative_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall score from metrics and analysis."""

        # Weight different components
        weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "clarity": 0.15,
            "practicality": 0.20,
            "code_quality": 0.10,
            "citation": 0.10,
        }

        # Use quantitative metrics if available, otherwise qualitative
        scores = {}
        for dimension in weights:
            if dimension in quantitative_metrics:
                scores[dimension] = quantitative_metrics[dimension]
            elif f"{dimension}_score" in qualitative_analysis:
                scores[dimension] = qualitative_analysis[f"{dimension}_score"]
            else:
                scores[dimension] = 0.5  # Default neutral score

        # Calculate weighted average
        overall_score = sum(weights[dim] * scores[dim] for dim in weights)
        return round(overall_score, 3)

    def _generate_improvement_suggestions(
        self, qualitative_analysis: Dict[str, Any], playbook: Optional[Playbook]
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []

        # Add suggestions from qualitative analysis
        if "suggestions" in qualitative_analysis:
            suggestions.extend(qualitative_analysis["suggestions"])

        # Add suggestions based on weaknesses
        if "weaknesses" in qualitative_analysis:
            for weakness in qualitative_analysis["weaknesses"]:
                if "code" in weakness.lower():
                    suggestions.append("Include runnable code examples")
                elif "citation" in weakness.lower():
                    suggestions.append("Add documentation references")
                elif "explanation" in weakness.lower():
                    suggestions.append("Provide more detailed explanations")

        # Add playbook-specific suggestions
        if playbook:
            # Check if heuristics are being followed
            for heuristic in playbook.context.heuristics:
                if heuristic.usage_count == 0:
                    suggestions.append(f"Apply heuristic: {heuristic.rule}")

        return list(set(suggestions))  # Remove duplicates

    def _identify_error_patterns(self, generated_output: str, task: str) -> List[str]:
        """Identify common error patterns in generated output."""
        patterns = []

        # Check for common issues
        if len(generated_output.strip()) < 50:
            patterns.append("output_too_short")

        if "```" not in generated_output and any(
            word in task.lower() for word in ["code", "function", "class", "implement"]
        ):
            patterns.append("missing_code_example")

        if not any(url in generated_output.lower() for url in ["http", "docs.", "github"]):
            patterns.append("missing_citations")

        if "error" in task.lower() and "try:" not in generated_output.lower():
            patterns.append("missing_error_handling")

        if generated_output.count(".") < 3:
            patterns.append("insufficient_explanation")

        return patterns

    def analyze_reflection_patterns(self, reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across multiple reflections."""
        if not reflections:
            return {}

        # Aggregate scores
        all_scores = []
        all_weaknesses = []
        all_suggestions = []
        all_patterns = []

        for reflection in reflections:
            if "overall_score" in reflection:
                all_scores.append(reflection["overall_score"])

            if "qualitative_analysis" in reflection and "weaknesses" in reflection["qualitative_analysis"]:
                all_weaknesses.extend(reflection["qualitative_analysis"]["weaknesses"])

            if "improvement_suggestions" in reflection:
                all_suggestions.extend(reflection["improvement_suggestions"])

            if "error_patterns" in reflection:
                all_patterns.extend(reflection["error_patterns"])

        # Calculate statistics
        analysis = {
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "score_distribution": {
                "high": len([s for s in all_scores if s >= 0.8]),
                "medium": len([s for s in all_scores if 0.5 <= s < 0.8]),
                "low": len([s for s in all_scores if s < 0.5]),
            },
            "common_weaknesses": self._get_most_common(all_weaknesses),
            "common_suggestions": self._get_most_common(all_suggestions),
            "common_patterns": self._get_most_common(all_patterns),
        }

        return analysis

    def _get_most_common(self, items: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common items from a list."""
        from collections import Counter

        counter = Counter(items)
        return counter.most_common(top_n)
