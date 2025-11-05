"""
ACE Runner - Main iteration controller for the ACE system.
Manages full iteration cycles with checkpointing, convergence detection, and evaluation.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..eval import DatasetGenerator, Evaluator
from ..playbook_schema import Playbook
from ..utils.config import ConfigManager
from ..utils.llm_client import LLMClient, LLMConfig
from ..utils.versioning import commit_playbook, tag_playbook_version
from .orchestrator import ACEOrchestrator

logger = logging.getLogger(__name__)


class ACERunner:
    """
    Main ACE iteration controller.

    Manages the complete ACE workflow including initialization,
    iteration execution, convergence detection, and result management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize ACE runner.

        Args:
            config: Configuration dictionary (optional)
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        if config:
            self.config = config
        else:
            config_manager = ConfigManager(config_path)
            self.config = config_manager.load_config()

        # Initialize components
        self.llm_config = LLMConfig(**self.config["llm"])
        self.llm_client = LLMClient(self.llm_config)
        # Try to preload evaluation dataset (dev set) for gating
        eval_dataset = None
        try:
            dev_path = Path(self.config["logging"]["output_dir"]) / "dev_set.json"
            if dev_path.exists():
                with open(dev_path) as f:
                    eval_dataset = json.load(f)
        except Exception:
            eval_dataset = None

        self.orchestrator = ACEOrchestrator(self.llm_client, evaluation_dataset=eval_dataset)
        # Wire thresholds and caps from config
        try:
            self.orchestrator.bleu_threshold = float(self.config["evaluation"].get("gate_bleu_delta", 0.01))
            self.orchestrator.em_threshold = float(self.config["evaluation"].get("gate_em_delta", 0.02))
        except Exception:
            pass
        try:
            self.orchestrator.curator.max_examples = int(self.config["ace"].get("max_examples", 8))
        except Exception:
            pass
        self.evaluator = Evaluator()
        self.dataset_generator = DatasetGenerator()

        # Setup output directory
        self.output_dir = Path(self.config["logging"]["output_dir"])
        self.output_dir.mkdir(exist_ok=True)

        logger.info("ACE runner initialized")

    def _find_latest_playbook(self) -> Optional[str]:
        """
        Find the latest playbook file in the output directory.

        Returns:
            Path to the latest playbook file, or None if none found
        """
        playbook_files = list(self.output_dir.glob("playbook_v*.yaml"))
        if not playbook_files:
            return None

        # Sort by modification time and return the latest
        latest_file = max(playbook_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Found latest playbook: {latest_file}")
        return str(latest_file)

    def run_iterations(
        self,
        num_iterations: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        playbook_path: Optional[str] = None,
        evaluation_dataset: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Run ACE iterations with the specified parameters.

        Args:
            num_iterations: Number of iterations to run (uses config default if None)
            tasks: List of tasks to process (generates synthetic if None)
            playbook_path: Path to initial playbook (uses base if None)

        Returns:
            Results of the ACE iterations
        """
        # Load or create initial playbook
        if playbook_path:
            playbook = Playbook.from_yaml(playbook_path)
            logger.info(f"Using specified playbook: {playbook_path}")
        else:
            # Try to find latest playbook, fallback to base
            latest_playbook = self._find_latest_playbook()
            if latest_playbook:
                playbook = Playbook.from_yaml(latest_playbook)
                logger.info(f"Using latest playbook: {latest_playbook}")
            else:
                playbook = Playbook.from_yaml("agentic_context_engineering/playbook_schema/base_playbook.yaml")
                logger.info("Using base playbook (no previous playbooks found)")

        # Generate or load tasks
        if tasks is None:
            tasks = self._generate_tasks()

        # Set number of iterations
        if num_iterations is None:
            num_iterations = self.config["ace"]["max_iterations"]

        logger.info(f"Starting ACE iterations: {num_iterations} iterations, {len(tasks)} tasks")

        # Run iterations
        start_time = time.time()
        # Ensure orchestrator has evaluation dataset; if not, try to load dev set here
        if not self.orchestrator.evaluation_dataset:
            dev_path = self.output_dir / "dev_set.json"
            if dev_path.exists():
                with open(dev_path) as f:
                    self.orchestrator.evaluation_dataset = json.load(f)

        results = self.orchestrator.run_iterations(
            playbook, tasks, num_iterations, evaluation_dataset=evaluation_dataset
        )

        # Calculate runtime
        runtime = time.time() - start_time

        # Add runtime and configuration info
        results["runtime_seconds"] = runtime
        results["configuration"] = {
            "max_iterations": num_iterations,
            "num_tasks": len(tasks),
            "llm_model": self.llm_config.model,
            "convergence_threshold": self.config["ace"]["convergence_threshold"],
        }

        # Save results
        self._save_results(results)

        # Commit playbook changes to git
        if results["final_playbook"]:
            self._commit_playbook(results["final_playbook"])

        logger.info(f"ACE iterations completed in {runtime:.2f} seconds")
        return results

    def run_validation(self, dev_iterations: int = 3) -> Dict[str, Any]:
        """Run a short dev iteration cycle, then evaluate on holdout set.
        Returns combined summary without executing long runs.
        """
        # Load or create playbook
        latest_playbook = self._find_latest_playbook()
        if latest_playbook:
            playbook = Playbook.from_yaml(latest_playbook)
        else:
            playbook = Playbook.from_yaml("agentic_context_engineering/playbook_schema/base_playbook.yaml")

        # Load dev/holdout datasets if present
        dev_path = self.output_dir / "dev_set.json"
        holdout_path = self.output_dir / "holdout_set.json"
        dev_tasks = None
        dev_dataset = None
        if dev_path.exists():
            with open(dev_path) as f:
                dev_dataset = json.load(f)
                dev_tasks = [s["input"] for s in dev_dataset]
        if dev_tasks is None:
            dev_tasks = self._generate_tasks()
            # regenerate paths after generation
            if holdout_path.exists():
                with open(holdout_path) as f:
                    holdout_dataset = json.load(f)
            else:
                holdout_dataset = self.dataset_generator.generate_dataset(size=10)
        else:
            if holdout_path.exists():
                with open(holdout_path) as f:
                    holdout_dataset = json.load(f)
            else:
                holdout_dataset = self.dataset_generator.generate_dataset(size=10)

        # Ensure orchestrator gating uses dev set
        self.orchestrator.evaluation_dataset = dev_dataset or []

        # Run a few iterations on dev tasks
        results = self.orchestrator.run_iterations(
            playbook, dev_tasks, dev_iterations, evaluation_dataset=self.orchestrator.evaluation_dataset
        )

        # Evaluate final playbook on holdout
        final_playbook = results.get("final_playbook", playbook)
        holdout_eval = self.evaluator.evaluate_playbook(final_playbook, holdout_dataset, self.orchestrator.generator)

        return {
            "dev_iterations": results,
            "holdout_evaluation": holdout_eval,
        }

    def _generate_tasks(self) -> List[str]:
        """Generate synthetic tasks for iteration."""
        # Load iteration dataset
        iteration_set_path = self.output_dir / "iteration_set.json"

        if iteration_set_path.exists():
            with open(iteration_set_path) as f:
                dataset = json.load(f)
        else:
            # Generate new dataset
            logger.info("Generating synthetic dataset")
            dataset = self.dataset_generator.generate_dataset(size=20)

            # Split dataset
            dev_set, iteration_set, holdout_set = self.dataset_generator.split_dataset(dataset)

            # Save datasets
            self.dataset_generator.save_dataset(dev_set, str(self.output_dir / "dev_set.json"))
            self.dataset_generator.save_dataset(iteration_set, str(self.output_dir / "iteration_set.json"))
            self.dataset_generator.save_dataset(holdout_set, str(self.output_dir / "holdout_set.json"))

        # Extract tasks
        tasks = [sample["input"] for sample in dataset]
        return tasks

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save iteration results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results
        results_path = self.output_dir / f"ace_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save final playbook
        if results["final_playbook"]:
            playbook_path = self.output_dir / f"playbook_v{results['final_playbook'].version}.yaml"
            results["final_playbook"].to_yaml(str(playbook_path))

        # Save metrics history
        if results["iterations"]:
            metrics_history = self._extract_metrics_history(results["iterations"])
            metrics_path = self.output_dir / f"metrics_history_{timestamp}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")

    def _extract_metrics_history(self, iterations: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract metrics history from iterations."""
        metrics_history = {
            "iteration": [],
            "avg_reflection_score": [],
            "avg_output_length": [],
            "code_example_rate": [],
            "citation_rate": [],
            "bleu": [],
            "exact_match": [],
            "rougeL": [],
            "semantic_similarity": [],
            "overall_accuracy": [],
            "delta_bleu": [],
            "delta_exact_match": [],
            "num_heuristics": [],
            "num_examples": [],
        }

        for iteration in iterations:
            if "metrics" in iteration:
                metrics = iteration["metrics"]
                metrics_history["iteration"].append(metrics.get("iteration", 0))
                metrics_history["avg_reflection_score"].append(metrics.get("avg_reflection_score", 0.0))
                metrics_history["avg_output_length"].append(metrics.get("avg_output_length", 0.0))
                metrics_history["code_example_rate"].append(metrics.get("code_example_rate", 0.0))
                metrics_history["citation_rate"].append(metrics.get("citation_rate", 0.0))
                metrics_history["bleu"].append(metrics.get("bleu", 0.0))
                metrics_history["exact_match"].append(metrics.get("exact_match", 0.0))
                metrics_history["rougeL"].append(metrics.get("rougeL", 0.0))
                metrics_history["semantic_similarity"].append(metrics.get("semantic_similarity", 0.0))
                metrics_history["overall_accuracy"].append(metrics.get("overall_accuracy", 0.0))
                metrics_history["delta_bleu"].append(metrics.get("delta_bleu", 0.0))
                metrics_history["delta_exact_match"].append(metrics.get("delta_exact_match", 0.0))
                metrics_history["num_heuristics"].append(metrics.get("playbook_heuristics", 0))
                metrics_history["num_examples"].append(metrics.get("playbook_examples", 0))

        return metrics_history

    def _commit_playbook(self, playbook: Playbook) -> None:
        """Commit playbook changes to git."""
        try:
            playbook_path = self.output_dir / f"playbook_v{playbook.version}.yaml"
            message = f"ACE iteration: Updated playbook to v{playbook.version}"

            if commit_playbook(str(playbook_path), playbook.version, message):
                logger.info(f"Playbook v{playbook.version} committed to git")

                # Create tag for significant versions
                if playbook.version.endswith(".0"):  # Major or minor version
                    tag_playbook_version(playbook.version, f"ACE playbook version {playbook.version}")
        except Exception as e:
            logger.warning(f"Failed to commit playbook: {e}")

    def evaluate_playbook(
        self, playbook: Playbook, test_dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a playbook against a test dataset.

        Args:
            playbook: Playbook to evaluate
            test_dataset: Test dataset (loads holdout set if None)

        Returns:
            Evaluation results
        """
        if test_dataset is None:
            # Load holdout dataset
            holdout_path = self.output_dir / "holdout_set.json"
            if holdout_path.exists():
                with open(holdout_path) as f:
                    test_dataset = json.load(f)
            else:
                logger.warning("No holdout dataset found, generating new test set")
                test_dataset = self.dataset_generator.generate_dataset(size=10)

        # Run evaluation
        evaluation_results = self.evaluator.evaluate_playbook(playbook, test_dataset, self.orchestrator.generator)

        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path = self.output_dir / f"evaluation_{timestamp}.json"
        self.evaluator.save_evaluation_results(evaluation_results, str(eval_path))

        return evaluation_results

    def compare_playbooks(
        self, playbook1_path: str, playbook2_path: str, test_dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare two playbook versions.

        Args:
            playbook1_path: Path to first playbook
            playbook2_path: Path to second playbook
            test_dataset: Test dataset for comparison

        Returns:
            Comparison results
        """
        # Load playbooks
        playbook1 = Playbook.from_yaml(playbook1_path)
        playbook2 = Playbook.from_yaml(playbook2_path)

        # Load test dataset
        if test_dataset is None:
            holdout_path = self.output_dir / "holdout_set.json"
            if holdout_path.exists():
                with open(holdout_path) as f:
                    test_dataset = json.load(f)
            else:
                test_dataset = self.dataset_generator.generate_dataset(size=10)

        # Run comparison
        comparison_results = self.evaluator.compare_playbooks(
            playbook1, playbook2, test_dataset, self.orchestrator.generator
        )

        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = self.output_dir / f"comparison_{timestamp}.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)

        return comparison_results

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "llm_client": {
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "health_check": self.llm_client.health_check(),
                "vram_usage": self.llm_client.get_vram_usage(),
            },
            "orchestrator": self.orchestrator.get_workflow_status(),
            "output_directory": str(self.output_dir),
            "configuration": {
                "max_iterations": self.config["ace"]["max_iterations"],
                "convergence_threshold": self.config["ace"]["convergence_threshold"],
                "reflection_batch_size": self.config["ace"]["reflection_batch_size"],
            },
        }

    def warm_up(self) -> None:
        """Warm up the system components."""
        logger.info("Warming up ACE system...")

        # Warm up LLM client
        self.llm_client.warm_up()

        # Test orchestrator
        test_playbook = Playbook.from_yaml("agentic_context_engineering/playbook_schema/base_playbook.yaml")
        test_tasks = ["What is Python?"]

        try:
            result = self.orchestrator.run_iteration(test_playbook, test_tasks, 0)
            if result["success"]:
                logger.info("✓ System warm-up successful")
            else:
                logger.warning(f"System warm-up had issues: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"System warm-up failed: {e}")
            raise
