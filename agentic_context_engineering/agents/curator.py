"""
Curator Agent for ACE framework.
Updates playbook based on reflection feedback and maintains context evolution.
"""

import logging
from typing import Any, Dict, List, Optional

from ..playbook_schema import Heuristic, HistoryEntry, Playbook
from ..utils.llm_client import LLMClient
from ..utils.metrics import MetricsCalculator
from ..utils.versioning import increment_version, suggest_version_bump

logger = logging.getLogger(__name__)


class Curator:
    """
    Curator agent that updates the playbook based on reflection feedback.

    The curator analyzes reflection results, identifies improvement opportunities,
    and updates the playbook with new heuristics, examples, and instructions.
    """

    def __init__(self, llm_client: LLMClient, max_examples: int = 8):
        """
        Initialize Curator agent.

        Args:
            llm_client: LLM client for curation analysis
        """
        self.llm_client = llm_client
        self.max_examples = max_examples
        logger.info("Curator agent initialized")

    def curate(self, playbook: Playbook, reflections: List[Dict[str, Any]], iteration: int) -> Playbook:
        """
        Update playbook based on reflection feedback.

        Args:
            playbook: Current playbook
            reflections: List of reflection results
            iteration: Current iteration number

        Returns:
            Updated playbook
        """
        # Create a copy to avoid modifying original
        updated_playbook = playbook.model_copy(deep=True)

        # Analyze reflection patterns
        analysis = self._analyze_reflections(reflections)

        # Update playbook based on analysis
        changes_made = []

        # Update heuristics
        heuristic_changes = self._update_heuristics(updated_playbook, analysis, iteration)
        changes_made.extend(heuristic_changes)

        # Add new examples
        example_changes = self._add_examples(updated_playbook, reflections, iteration)
        changes_made.extend(example_changes)

        # Example deduplication and pruning
        self._deduplicate_examples(updated_playbook)
        self._prune_examples(updated_playbook, max_examples=self.max_examples)

        # Update system instructions if needed
        instruction_changes = self._update_instructions(updated_playbook, analysis, iteration)
        changes_made.extend(instruction_changes)

        # Quality guards before metadata/versioning
        self._deduplicate_heuristics(updated_playbook)
        self._clean_repetitions_in_heuristics(updated_playbook)

        # Update metadata
        self._update_metadata(updated_playbook, iteration, changes_made)

        # Determine version bump
        change_types = [change["type"] for change in changes_made]
        bump_type = suggest_version_bump(change_types)
        updated_playbook.version = increment_version(playbook.version, bump_type)

        logger.info(f"Playbook updated: {len(changes_made)} changes, version {updated_playbook.version}")
        return updated_playbook

    def _analyze_reflections(self, reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reflection patterns to identify improvement opportunities."""
        analysis = {
            "common_weaknesses": [],
            "common_suggestions": [],
            "score_trends": [],
            "error_patterns": [],
            "improvement_areas": [],
        }

        all_weaknesses = []
        all_suggestions = []
        all_scores = []
        all_patterns = []

        for reflection in reflections:
            # Collect weaknesses
            if "qualitative_analysis" in reflection and "weaknesses" in reflection["qualitative_analysis"]:
                all_weaknesses.extend(reflection["qualitative_analysis"]["weaknesses"])

            # Collect suggestions
            if "improvement_suggestions" in reflection:
                all_suggestions.extend(reflection["improvement_suggestions"])

            # Collect scores
            if "overall_score" in reflection:
                all_scores.append(reflection["overall_score"])

            # Collect error patterns
            if "error_patterns" in reflection:
                all_patterns.extend(reflection["error_patterns"])

        # Analyze patterns
        from collections import Counter

        analysis["common_weaknesses"] = Counter(all_weaknesses).most_common(5)
        analysis["common_suggestions"] = Counter(all_suggestions).most_common(5)
        analysis["score_trends"] = {
            "average": sum(all_scores) / len(all_scores) if all_scores else 0,
            "min": min(all_scores) if all_scores else 0,
            "max": max(all_scores) if all_scores else 0,
        }
        analysis["error_patterns"] = Counter(all_patterns).most_common(5)

        # Identify improvement areas
        analysis["improvement_areas"] = self._identify_improvement_areas(analysis)

        return analysis

    def _identify_improvement_areas(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify specific areas that need improvement."""
        areas = []

        # Check for common weaknesses
        for weakness, count in analysis["common_weaknesses"]:
            if "code" in weakness.lower() and count >= 2:
                areas.append("code_examples")
            elif "citation" in weakness.lower() and count >= 2:
                areas.append("documentation_citations")
            elif "explanation" in weakness.lower() and count >= 2:
                areas.append("detailed_explanations")

        # Check for error patterns
        for pattern, count in analysis["error_patterns"]:
            if pattern == "missing_code_example" and count >= 2:
                areas.append("code_examples")
            elif pattern == "missing_citations" and count >= 2:
                areas.append("documentation_citations")
            elif pattern == "insufficient_explanation" and count >= 2:
                areas.append("detailed_explanations")

        return list(set(areas))  # Remove duplicates

    def _update_heuristics(self, playbook: Playbook, analysis: Dict[str, Any], iteration: int) -> List[Dict[str, Any]]:
        """Update heuristics based on analysis."""
        changes = []

        # Generate new heuristics for improvement areas
        for area in analysis["improvement_areas"]:
            new_heuristic = self._generate_heuristic_for_area(area, analysis)
            if new_heuristic:
                heuristic_id = playbook.add_heuristic(
                    rule=new_heuristic["rule"], confidence=new_heuristic["confidence"], iteration=iteration
                )
                changes.append({"type": "heuristic_added", "heuristic_id": heuristic_id, "rule": new_heuristic["rule"]})

        # Update existing heuristics based on common suggestions
        for suggestion, count in analysis["common_suggestions"]:
            if count >= 2:  # If suggestion appears multiple times
                updated = self._update_heuristic_for_suggestion(playbook, suggestion, iteration)
                if updated:
                    changes.append({
                        "type": "heuristic_updated",
                        "description": f"Updated based on suggestion: {suggestion}",
                    })

        return changes

    def _generate_heuristic_for_area(self, area: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a new heuristic for a specific improvement area."""

        heuristic_templates = {
            "code_examples": {
                "rule": "Always include runnable code examples for technical questions",
                "confidence": 0.85,
            },
            "documentation_citations": {
                "rule": "Cite official documentation with URLs for library/API questions",
                "confidence": 0.80,
            },
            "detailed_explanations": {
                "rule": "Provide step-by-step explanations for complex procedures",
                "confidence": 0.75,
            },
            "error_handling": {"rule": "Include error handling examples with try-except blocks", "confidence": 0.80},
        }

        return heuristic_templates.get(area)

    def _update_heuristic_for_suggestion(self, playbook: Playbook, suggestion: str, iteration: int) -> bool:
        """Update existing heuristics based on suggestions."""
        # Find relevant heuristics to update
        for heuristic in playbook.context.heuristics:
            if self._is_heuristic_relevant_to_suggestion(heuristic, suggestion):
                # Update the heuristic
                updated_rule = self._refine_heuristic_rule(heuristic.rule, suggestion)
                new_confidence = min(heuristic.confidence + 0.1, 1.0)  # Clamp to 1.0
                return playbook.update_heuristic(heuristic.id, updated_rule, new_confidence, iteration)

        return False

    def _is_heuristic_relevant_to_suggestion(self, heuristic: Heuristic, suggestion: str) -> bool:
        """Check if a heuristic is relevant to a suggestion."""
        suggestion_lower = suggestion.lower()
        rule_lower = heuristic.rule.lower()

        # Simple keyword matching
        return (
            ("code" in suggestion_lower and "code" in rule_lower)
            or ("citation" in suggestion_lower and "citation" in rule_lower)
            or ("explanation" in suggestion_lower and "explanation" in rule_lower)
        )

    def _refine_heuristic_rule(self, current_rule: str, suggestion: str) -> str:
        """Refine a heuristic rule based on a suggestion."""
        # Simple refinement - in practice, this could use LLM
        if "more specific" in suggestion.lower():
            return current_rule + " Be specific and detailed."
        elif "examples" in suggestion.lower():
            return current_rule + " Include practical examples."
        else:
            return current_rule

    def _add_examples(
        self, playbook: Playbook, reflections: List[Dict[str, Any]], iteration: int
    ) -> List[Dict[str, Any]]:
        """Add new few-shot examples based on good responses."""
        changes = []

        # Find high-quality responses to use as examples
        good_examples = []
        for reflection in reflections:
            if (
                reflection.get("overall_score", 0) >= 0.8
                and "qualitative_analysis" in reflection
                and "strengths" in reflection["qualitative_analysis"]
            ):
                good_examples.append({
                    "task": reflection["task"],
                    "output": reflection["generated_output"],
                    "strengths": reflection["qualitative_analysis"]["strengths"],
                })

        # Add up to 2 new examples
        for example in good_examples[:2]:
            annotation = f"Good example: {', '.join(example['strengths'][:2])}"
            playbook.add_example(
                input_text=example["task"],
                output_text=example["output"],
                annotation=annotation,
                quality_score=0.9,
                iteration=iteration,
            )
            changes.append({"type": "example_added", "description": "Added example from high-quality response"})

        return changes

    def _normalize_rule(self, rule: str) -> str:
        """Normalize a heuristic rule by collapsing repeated sentences and whitespace."""
        sentences = [s.strip() for s in rule.replace("\n", " ").split(".")]
        seen = set()
        cleaned: List[str] = []
        for s in sentences:
            if not s:
                continue
            key = s.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(s)
        return ". ".join(cleaned).strip() + ("." if cleaned else "")

    def _clean_repetitions_in_heuristics(self, playbook: Playbook) -> None:
        """Remove repeated phrases and normalize heuristic rules in-place."""
        for h in playbook.context.heuristics:
            h.rule = self._normalize_rule(h.rule)

    def _deduplicate_heuristics(self, playbook: Playbook) -> None:
        """Deduplicate heuristics by normalized rule, keeping highest confidence and latest update."""
        normalized_to_best = {}
        for h in playbook.context.heuristics:
            norm = self._normalize_rule(h.rule)
            if norm in normalized_to_best:
                existing = normalized_to_best[norm]
                # Prefer higher confidence, then later last_updated
                if (h.confidence, h.last_updated) > (existing.confidence, existing.last_updated):
                    normalized_to_best[norm] = h
            else:
                normalized_to_best[norm] = h
        # Replace list with unique best heuristics
        playbook.context.heuristics = list(normalized_to_best.values())

    def _deduplicate_examples(self, playbook: Playbook, similarity_threshold: float = 0.92) -> None:
        """Remove near-duplicate few-shot examples using semantic similarity on input+output."""
        examples = playbook.context.few_shot_examples
        if not examples:
            return
        calc = MetricsCalculator()
        if calc.sentence_model is None:
            # Fallback: simple text-key dedup
            seen = set()
            unique = []
            for ex in examples:
                key = (ex.input.strip() + " || " + ex.output.strip()).lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(ex)
            playbook.context.few_shot_examples = unique
            return

        texts = [f"{ex.input}\n\n{ex.output}" for ex in examples]
        embeddings = calc.sentence_model.encode(texts)
        kept_indices: List[int] = []
        for i, emb_i in enumerate(embeddings):
            is_dup = False
            for idx in kept_indices:
                emb_j = embeddings[idx]
                sim = float((emb_i @ emb_j) / ((emb_i**2).sum() ** 0.5 * (emb_j**2).sum() ** 0.5))
                if sim >= similarity_threshold:
                    # Keep the higher-quality example
                    if examples[i].quality_score > examples[idx].quality_score:
                        kept_indices.remove(idx)
                        kept_indices.append(i)
                    is_dup = True
                    break
            if not is_dup:
                kept_indices.append(i)
        playbook.context.few_shot_examples = [examples[i] for i in kept_indices]

    def _prune_examples(self, playbook: Playbook, max_examples: int = 8) -> None:
        """Prune examples to a maximum size using greedy diversity (max-min) by embeddings and quality."""
        examples = playbook.context.few_shot_examples
        if len(examples) <= max_examples:
            return
        calc = MetricsCalculator()
        texts = [f"{ex.input}\n\n{ex.output}" for ex in examples]
        if calc.sentence_model is None:
            # Fallback: highest quality first
            playbook.context.few_shot_examples = sorted(examples, key=lambda e: e.quality_score, reverse=True)[
                :max_examples
            ]
            return
        embs = calc.sentence_model.encode(texts)
        # Seed with highest quality
        seed_idx = max(range(len(examples)), key=lambda i: examples[i].quality_score)
        selected = [seed_idx]
        while len(selected) < max_examples:
            best_idx = None
            best_score = -1.0
            for i in range(len(examples)):
                if i in selected:
                    continue
                # max-min distance to selected set
                min_sim = 1.0
                for j in selected:
                    sim = float((embs[i] @ embs[j]) / ((embs[i] ** 2).sum() ** 0.5 * (embs[j] ** 2).sum() ** 0.5))
                    if sim < min_sim:
                        min_sim = sim
                # combine diversity and quality
                score = (1.0 - min_sim) * 0.7 + examples[i].quality_score * 0.3
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)
        playbook.context.few_shot_examples = [examples[i] for i in selected]

    def _update_instructions(
        self, playbook: Playbook, analysis: Dict[str, Any], iteration: int
    ) -> List[Dict[str, Any]]:
        """Update system instructions if needed."""
        changes = []

        # Check if instructions need updating based on common weaknesses
        needs_update = False
        new_instructions = playbook.context.system_instructions

        # Add guidance for common improvement areas
        if "code_examples" in analysis["improvement_areas"] and "code examples" not in new_instructions.lower():
            new_instructions += "\nAlways provide runnable code examples when relevant."
            needs_update = True

        if "documentation_citations" in analysis["improvement_areas"] and "citation" not in new_instructions.lower():
            new_instructions += "\nCite official documentation sources with URLs."
            needs_update = True

        if needs_update:
            playbook.context.system_instructions = new_instructions
            changes.append({
                "type": "instruction_updated",
                "description": "Updated system instructions based on common weaknesses",
            })

        return changes

    def _update_metadata(self, playbook: Playbook, iteration: int, changes: List[Dict[str, Any]]):
        """Update playbook metadata."""
        playbook.metadata.iteration = iteration
        playbook.metadata.parent_version = playbook.version

        # Update convergence status based on changes
        if len(changes) == 0:
            playbook.metadata.convergence_status = "plateaued"
        elif any("heuristic_added" in change["type"] for change in changes):
            playbook.metadata.convergence_status = "improving"
        else:
            playbook.metadata.convergence_status = "improving"

        # Add history entries
        for change in changes:
            history_entry = HistoryEntry(
                iteration=iteration,
                change_type=change["type"],
                description=change.get("description", f"Change: {change['type']}"),
            )
            playbook.history.append(history_entry)

    def analyze_playbook_evolution(self, playbooks: List[Playbook]) -> Dict[str, Any]:
        """Analyze how the playbook has evolved over iterations."""
        if len(playbooks) < 2:
            return {}

        initial = playbooks[0]
        final = playbooks[-1]

        evolution = {
            "version_progression": [p.version for p in playbooks],
            "heuristic_growth": len(final.context.heuristics) - len(initial.context.heuristics),
            "example_growth": len(final.context.few_shot_examples) - len(initial.context.few_shot_examples),
            "instruction_changes": final.context.system_instructions != initial.context.system_instructions,
            "performance_trend": final.metadata.performance_metrics.accuracy
            - initial.metadata.performance_metrics.accuracy,
            "convergence_status": final.metadata.convergence_status,
        }

        # Analyze heuristic evolution
        evolution["heuristic_evolution"] = {
            "new_heuristics": [h for h in final.context.heuristics if h.created_iteration > 0],
            "updated_heuristics": [h for h in final.context.heuristics if h.last_updated > h.created_iteration],
            "high_confidence_heuristics": [h for h in final.context.heuristics if h.confidence >= 0.8],
        }

        return evolution
