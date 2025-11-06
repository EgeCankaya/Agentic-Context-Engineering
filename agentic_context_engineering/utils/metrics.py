"""
Metrics calculation for ACE evaluation.
Implements BLEU, ROUGE, exact match, and semantic similarity metrics.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from sentence_transformers import SentenceTransformer


class MetricsCalculator:
    """Calculator for various evaluation metrics."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize metrics calculator.

        Args:
            model_name: Sentence transformer model for semantic similarity
        """
        self.bleu = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None

    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            BLEU score (0-1)
        """
        if not predictions or not references:
            return 0.0

        # Ensure same length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

        try:
            bleu_score = self.bleu.corpus_score(predictions, [references])
            return bleu_score.score / 100.0  # Convert to 0-1 scale
        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            return 0.0

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not predictions or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        # Ensure same length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores["rouge1"] += scores["rouge1"].fmeasure
            rouge_scores["rouge2"] += scores["rouge2"].fmeasure
            rouge_scores["rougeL"] += scores["rougeL"].fmeasure

        # Average scores
        rouge_scores = {k: v / len(predictions) for k, v in rouge_scores.items()}
        return rouge_scores

    def calculate_exact_match(
        self, predictions: Union[Sequence[str], str], references: Union[Sequence[str], str]
    ) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Exact match accuracy (0-1)
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        if not predictions or not references:
            return 0.0

        # Ensure same length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1

        return float(exact_matches / len(predictions))

    def calculate_semantic_similarity(
        self, predictions: Union[Sequence[str], str], references: Union[Sequence[str], str]
    ) -> float:
        """
        Calculate semantic similarity using sentence transformers.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Average semantic similarity (0-1)
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        if not predictions or not references or self.sentence_model is None:
            return 0.0

        # Ensure same length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

        try:
            # Encode texts
            pred_embeddings = self.sentence_model.encode(predictions)
            ref_embeddings = self.sentence_model.encode(references)

            # Calculate cosine similarity
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                similarity = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
                similarities.append(similarity)

            return float(np.mean(similarities))
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def calculate_custom_accuracy(
        self, predictions: List[str], references: List[str], criteria: List[str]
    ) -> Dict[str, float]:
        """
        Calculate custom accuracy based on specific criteria.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            criteria: List of criteria to check (e.g., ["code_example", "citation", "explanation"])

        Returns:
            Dictionary with accuracy for each criterion
        """
        if not predictions or not references:
            return dict.fromkeys(criteria, 0.0)

        # Ensure same length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

        criterion_scores = dict.fromkeys(criteria, 0.0)

        for pred, ref in zip(predictions, references):
            for criterion in criteria:
                if self._check_criterion(pred, criterion):
                    criterion_scores[criterion] += 1

        # Normalize by number of samples
        criterion_scores = {k: v / len(predictions) for k, v in criterion_scores.items()}
        return criterion_scores

    def _check_criterion(self, text: str, criterion: str) -> bool:
        """Check if text meets a specific criterion."""
        text_lower = text.lower()

        if criterion == "code_example":
            # Check for code blocks or Python keywords
            return (
                "```" in text
                or "def " in text_lower
                or "import " in text_lower
                or "class " in text_lower
                or "if " in text_lower
            )

        elif criterion == "citation":
            # Check for URLs or documentation references
            return (
                "http" in text_lower
                or "docs." in text_lower
                or "documentation" in text_lower
                or "readthedocs" in text_lower
            )

        elif criterion == "explanation":
            # Check for explanatory words
            return any(
                word in text_lower
                for word in ["because", "since", "therefore", "however", "additionally", "furthermore"]
            )

        elif criterion == "error_handling":
            # Check for error handling patterns
            return "try:" in text_lower or "except" in text_lower or "raise" in text_lower or "error" in text_lower

        else:
            return False

    def calculate_all_metrics(
        self, predictions: List[str], references: List[str], custom_criteria: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            custom_criteria: Optional list of custom criteria

        Returns:
            Dictionary with all metric scores
        """
        metrics = {}

        # Standard metrics
        metrics["bleu"] = self.calculate_bleu(predictions, references)

        rouge_scores = self.calculate_rouge(predictions, references)
        metrics.update(rouge_scores)

        metrics["exact_match"] = self.calculate_exact_match(predictions, references)
        metrics["semantic_similarity"] = self.calculate_semantic_similarity(predictions, references)

        # Custom criteria
        if custom_criteria:
            custom_scores = self.calculate_custom_accuracy(predictions, references, custom_criteria)
            metrics.update(custom_scores)

        return metrics

    def calculate_improvement(
        self, before_metrics: Dict[str, float], after_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate improvement between two metric sets.

        Args:
            before_metrics: Metrics from before
            after_metrics: Metrics from after

        Returns:
            Dictionary with improvement percentages
        """
        improvements = {}

        for metric in before_metrics:
            if metric in after_metrics:
                before_val = before_metrics[metric]
                after_val = after_metrics[metric]

                if before_val > 0:
                    improvement = ((after_val - before_val) / before_val) * 100
                else:
                    improvement = 0.0 if after_val == 0 else 100.0

                improvements[f"{metric}_improvement"] = improvement

        return improvements


# Example usage
if __name__ == "__main__":
    calculator = MetricsCalculator()

    predictions = [
        "Use the json module to read JSON files in Python.",
        "Here's how to handle errors with try-except blocks.",
    ]
    references = [
        "Use the json module from the standard library to read JSON files.",
        "Use try-except blocks to handle errors gracefully in Python.",
    ]

    metrics = calculator.calculate_all_metrics(
        predictions, references, custom_criteria=["code_example", "citation", "explanation"]
    )

    print("Metrics:", metrics)
