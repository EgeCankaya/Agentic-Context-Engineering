"""
Evaluator for ACE system.
Handles comprehensive evaluation of playbook performance across multiple metrics.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..playbook_schema import Playbook
from ..utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for ACE system performance.

    Evaluates playbook performance across multiple dimensions including
    quantitative metrics, qualitative assessment, and task-specific accuracy.
    """

    def __init__(self, metrics_calculator: Optional[MetricsCalculator] = None):
        """
        Initialize evaluator.

        Args:
            metrics_calculator: Optional metrics calculator instance
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        logger.info("Evaluator initialized")

    def evaluate_playbook(
        self, playbook: Playbook, test_dataset: List[Dict[str, Any]], generator, num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a playbook against a test dataset.

        Args:
            playbook: Playbook to evaluate
            test_dataset: Test dataset with tasks and reference outputs
            generator: Generator agent for producing outputs
            num_samples: Optional limit on number of samples to evaluate

        Returns:
            Comprehensive evaluation results
        """
        if num_samples:
            test_dataset = test_dataset[:num_samples]

        logger.info(f"Evaluating playbook {playbook.version} on {len(test_dataset)} samples")

        # Generate outputs for all test samples
        tasks = [sample["input"] for sample in test_dataset]
        reference_outputs = [sample["reference_output"] for sample in test_dataset]

        # Generate responses using the playbook
        generated_outputs = generator.generate_batch(tasks, playbook)

        # Calculate quantitative metrics
        quantitative_metrics = self.metrics_calculator.calculate_all_metrics(
            generated_outputs,
            reference_outputs,
            custom_criteria=["code_example", "citation", "explanation", "error_handling"],
        )

        # Calculate task-specific accuracy
        task_accuracy = self._calculate_task_accuracy(test_dataset, generated_outputs)

        # Calculate response quality metrics
        quality_metrics = self._calculate_quality_metrics(generated_outputs, test_dataset)

        # Generate evaluation summary
        evaluation_summary = self._generate_evaluation_summary(
            quantitative_metrics, task_accuracy, quality_metrics, playbook
        )

        # Create before/after examples for visualization
        before_after_examples = self._create_before_after_examples(test_dataset, generated_outputs)

        return {
            "playbook_version": playbook.version,
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_samples": len(test_dataset),
            "quantitative_metrics": quantitative_metrics,
            "task_accuracy": task_accuracy,
            "quality_metrics": quality_metrics,
            "evaluation_summary": evaluation_summary,
            "before_after_examples": before_after_examples,
            "detailed_results": self._create_detailed_results(test_dataset, generated_outputs),
        }

    def apply_metrics_to_playbook(
        self,
        playbook: Playbook,
        quantitative_metrics: Dict[str, float],
        task_accuracy: Optional[Dict[str, float]] = None,
        avg_tokens: Optional[int] = None,
        inference_time_sec: Optional[float] = None,
        vram_usage_gb: Optional[float] = None,
    ) -> Playbook:
        """
        Write aggregated evaluation metrics into playbook.metadata.performance_metrics.

        Args:
            playbook: Playbook to update
            quantitative_metrics: Metrics dict from calculate_all_metrics (bleu, rouge*, exact_match, semantic_similarity)
            task_accuracy: Optional dict containing overall_accuracy
            avg_tokens: Optional average token usage for responses
            inference_time_sec: Optional average inference time in seconds
            vram_usage_gb: Optional VRAM usage in GB

        Returns:
            Updated playbook (same instance passed in)
        """
        pm = playbook.metadata.performance_metrics
        # Set primary metrics
        if task_accuracy is not None:
            pm.accuracy = float(task_accuracy.get("overall_accuracy", pm.accuracy))
        # BLEU
        if "bleu" in quantitative_metrics:
            pm.bleu_score = float(quantitative_metrics.get("bleu", 0.0))
        # Map ROUGE-L to rouge_score (single slot in schema)
        if "rougeL" in quantitative_metrics:
            pm.rouge_score = float(quantitative_metrics.get("rougeL", 0.0))
        # Exact match and semantic similarity
        if "exact_match" in quantitative_metrics:
            pm.exact_match = float(quantitative_metrics.get("exact_match", 0.0))
        if "semantic_similarity" in quantitative_metrics:
            pm.semantic_similarity = float(quantitative_metrics.get("semantic_similarity", 0.0))

        # Runtime/resource stats if provided
        if avg_tokens is not None:
            pm.avg_tokens = int(avg_tokens)
        if inference_time_sec is not None:
            pm.inference_time_sec = float(inference_time_sec)
        if vram_usage_gb is not None:
            pm.vram_usage_gb = float(vram_usage_gb)

        return playbook

    def _calculate_task_accuracy(
        self, test_dataset: List[Dict[str, Any]], generated_outputs: List[str]
    ) -> Dict[str, float]:
        """Calculate task-specific accuracy metrics."""
        accuracy_metrics = {}

        # Group by difficulty
        difficulty_groups = {"easy": [], "medium": [], "hard": []}
        for i, sample in enumerate(test_dataset):
            difficulty = sample.get("difficulty", "medium")
            if difficulty in difficulty_groups:
                difficulty_groups[difficulty].append((sample, generated_outputs[i]))

        # Calculate accuracy by difficulty
        for difficulty, samples in difficulty_groups.items():
            if samples:
                accuracy = self._calculate_accuracy_for_group(samples)
                accuracy_metrics[f"{difficulty}_accuracy"] = accuracy

        # Calculate overall accuracy
        all_samples = list(zip(test_dataset, generated_outputs))
        accuracy_metrics["overall_accuracy"] = self._calculate_accuracy_for_group(all_samples)

        return accuracy_metrics

    def _calculate_accuracy_for_group(self, samples: List[Tuple[Dict, str]]) -> float:
        """Calculate accuracy for a group of samples."""
        correct = 0
        total = len(samples)

        for sample, generated in samples:
            if self._is_response_correct(sample, generated):
                correct += 1

        return correct / total if total > 0 else 0.0

    def _is_response_correct(self, sample: Dict[str, Any], generated: str) -> bool:
        """Determine if a generated response is correct for the task."""
        criteria = sample.get("evaluation_criteria", {})
        generated_lower = generated.lower()

        # Check each criterion
        for criterion, description in criteria.items():
            if criterion == "code_quality" and "code example" in description.lower():
                if "```" not in generated and any(
                    word in sample["input"].lower() for word in ["code", "function", "implement"]
                ):
                    return False
            elif criterion == "documentation" and "citation" in description.lower():
                if not any(url in generated_lower for url in ["http", "docs.", "github"]):
                    return False
            elif criterion == "explanation" and "explanation" in description.lower():
                if len(generated.split(".")) < 3:  # Very basic explanation check
                    return False

        return True

    def _calculate_quality_metrics(
        self, generated_outputs: List[str], test_dataset: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate quality metrics for generated outputs."""
        metrics = {
            "avg_length": sum(len(output) for output in generated_outputs) / len(generated_outputs),
            "avg_sentences": sum(len(output.split(".")) for output in generated_outputs) / len(generated_outputs),
            "code_example_rate": sum(1 for output in generated_outputs if "```" in output) / len(generated_outputs),
            "citation_rate": sum(
                1 for output in generated_outputs if any(url in output.lower() for url in ["http", "docs."])
            )
            / len(generated_outputs),
            "structure_rate": sum(
                1 for output in generated_outputs if any(marker in output for marker in ["##", "-", "1.", "2."])
            )
            / len(generated_outputs),
        }

        return metrics

    def _generate_evaluation_summary(
        self,
        quantitative_metrics: Dict[str, float],
        task_accuracy: Dict[str, float],
        quality_metrics: Dict[str, float],
        playbook: Playbook,
    ) -> Dict[str, Any]:
        """Generate a comprehensive evaluation summary."""
        summary = {
            "overall_performance": {
                "bleu_score": quantitative_metrics.get("bleu", 0.0),
                "rouge1_score": quantitative_metrics.get("rouge1", 0.0),
                "exact_match": quantitative_metrics.get("exact_match", 0.0),
                "semantic_similarity": quantitative_metrics.get("semantic_similarity", 0.0),
            },
            "task_performance": task_accuracy,
            "quality_indicators": quality_metrics,
            "playbook_characteristics": {
                "num_heuristics": len(playbook.context.heuristics),
                "num_examples": len(playbook.context.few_shot_examples),
                "instruction_length": len(playbook.context.system_instructions),
                "convergence_status": playbook.metadata.convergence_status,
            },
            "performance_grade": self._calculate_performance_grade(quantitative_metrics, task_accuracy),
        }

        return summary

    def _calculate_performance_grade(
        self, quantitative_metrics: Dict[str, float], task_accuracy: Dict[str, float]
    ) -> str:
        """Calculate overall performance grade."""
        # Weight different metrics
        bleu_score = quantitative_metrics.get("bleu", 0.0)
        rouge_score = quantitative_metrics.get("rouge1", 0.0)
        exact_match = quantitative_metrics.get("exact_match", 0.0)
        overall_accuracy = task_accuracy.get("overall_accuracy", 0.0)

        # Calculate weighted average
        weighted_score = bleu_score * 0.2 + rouge_score * 0.2 + exact_match * 0.3 + overall_accuracy * 0.3

        if weighted_score >= 0.8:
            return "A"
        elif weighted_score >= 0.7:
            return "B"
        elif weighted_score >= 0.6:
            return "C"
        elif weighted_score >= 0.5:
            return "D"
        else:
            return "F"

    def _create_before_after_examples(
        self, test_dataset: List[Dict[str, Any]], generated_outputs: List[str]
    ) -> List[Dict[str, Any]]:
        """Create before/after examples for visualization."""
        examples = []

        # Select diverse examples
        sample_indices = [
            0,
            len(test_dataset) // 4,
            len(test_dataset) // 2,
            3 * len(test_dataset) // 4,
            len(test_dataset) - 1,
        ]

        for i in sample_indices:
            if i < len(test_dataset):
                sample = test_dataset[i]
                generated = generated_outputs[i]

                examples.append({
                    "input": sample["input"],
                    "reference": sample["reference_output"],
                    "generated": generated,
                    "difficulty": sample.get("difficulty", "medium"),
                    "tags": sample.get("tags", []),
                })

        return examples

    def _create_detailed_results(
        self, test_dataset: List[Dict[str, Any]], generated_outputs: List[str]
    ) -> List[Dict[str, Any]]:
        """Create detailed results for each sample."""
        detailed_results = []

        for i, (sample, generated) in enumerate(zip(test_dataset, generated_outputs)):
            result = {
                "sample_id": sample.get("id", f"sample_{i}"),
                "input": sample["input"],
                "reference_output": sample["reference_output"],
                "generated_output": generated,
                "difficulty": sample.get("difficulty", "medium"),
                "tags": sample.get("tags", []),
                "evaluation_criteria": sample.get("evaluation_criteria", {}),
                "is_correct": self._is_response_correct(sample, generated),
                "response_length": len(generated),
                "has_code": "```" in generated,
                "has_citation": any(url in generated.lower() for url in ["http", "docs."]),
            }
            detailed_results.append(result)

        return detailed_results

    def compare_playbooks(
        self, playbook1: Playbook, playbook2: Playbook, test_dataset: List[Dict[str, Any]], generator
    ) -> Dict[str, Any]:
        """
        Compare two playbook versions.

        Args:
            playbook1: First playbook version
            playbook2: Second playbook version
            test_dataset: Test dataset for evaluation
            generator: Generator agent

        Returns:
            Comparison results
        """
        logger.info(f"Comparing playbooks {playbook1.version} vs {playbook2.version}")

        # Evaluate both playbooks
        eval1 = self.evaluate_playbook(playbook1, test_dataset, generator)
        eval2 = self.evaluate_playbook(playbook2, test_dataset, generator)

        # Calculate improvements
        improvements = {}
        for metric in eval1["quantitative_metrics"]:
            if metric in eval2["quantitative_metrics"]:
                val1 = eval1["quantitative_metrics"][metric]
                val2 = eval2["quantitative_metrics"][metric]
                improvements[f"{metric}_improvement"] = val2 - val1

        # Calculate task accuracy improvements
        task_improvements = {}
        for metric in eval1["task_accuracy"]:
            if metric in eval2["task_accuracy"]:
                val1 = eval1["task_accuracy"][metric]
                val2 = eval2["task_accuracy"][metric]
                task_improvements[f"{metric}_improvement"] = val2 - val1

        return {
            "playbook1_version": playbook1.version,
            "playbook2_version": playbook2.version,
            "playbook1_results": eval1,
            "playbook2_results": eval2,
            "metric_improvements": improvements,
            "task_improvements": task_improvements,
            "overall_improvement": self._calculate_overall_improvement(improvements),
            "comparison_summary": self._generate_comparison_summary(improvements, task_improvements),
        }

    def _calculate_overall_improvement(self, improvements: Dict[str, float]) -> float:
        """Calculate overall improvement score."""
        if not improvements:
            return 0.0

        # Weight key metrics
        key_metrics = ["bleu_improvement", "rouge1_improvement", "exact_match_improvement"]
        weighted_improvements = []

        for metric in key_metrics:
            if metric in improvements:
                weighted_improvements.append(improvements[metric])

        return sum(weighted_improvements) / len(weighted_improvements) if weighted_improvements else 0.0

    def _generate_comparison_summary(
        self, metric_improvements: Dict[str, float], task_improvements: Dict[str, float]
    ) -> str:
        """Generate a summary of the comparison."""
        overall_improvement = self._calculate_overall_improvement(metric_improvements)

        if overall_improvement > 0.1:
            return f"Significant improvement (+{overall_improvement:.1%})"
        elif overall_improvement > 0.05:
            return f"Moderate improvement (+{overall_improvement:.1%})"
        elif overall_improvement > 0:
            return f"Minor improvement (+{overall_improvement:.1%})"
        elif overall_improvement > -0.05:
            return "No significant change"
        else:
            return f"Performance declined ({overall_improvement:.1%})"

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_evaluation_results(self, input_path: str) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        with open(input_path) as f:
            return json.load(f)
