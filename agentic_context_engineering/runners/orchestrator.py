"""
LangGraph orchestrator for ACE iteration cycles.
Manages the Generator → Reflector → Curator workflow.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents import Curator, Generator, Reflector
from ..eval.evaluator import Evaluator
from ..playbook_schema import Playbook
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ACEState(TypedDict):
    """State for ACE iteration workflow."""

    playbook: Playbook
    tasks: List[str]
    generated_outputs: List[str]
    reflections: List[Dict[str, Any]]
    updated_playbook: Playbook
    iteration: int
    metrics: Dict[str, Any]
    error: Optional[str]


class ACEOrchestrator:
    """
    LangGraph orchestrator for ACE iteration cycles.

    Manages the complete workflow: Generator → Reflector → Curator
    with state management and error handling.
    """

    def __init__(self, llm_client: LLMClient, evaluation_dataset: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize ACE orchestrator.

        Args:
            llm_client: LLM client for all agents
        """
        self.llm_client = llm_client
        self.generator = Generator(llm_client)
        self.reflector = Reflector(llm_client)
        self.curator = Curator(llm_client)
        self.evaluator = Evaluator()
        self.evaluation_dataset = evaluation_dataset
        # Gating thresholds (can be overridden by runner/config)
        self.bleu_threshold = 0.01
        self.em_threshold = 0.02

        # Create state graph
        self.graph = self._create_graph()

        logger.info("ACE orchestrator initialized")

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(ACEState)

        # Add nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("curator", self._curator_node)
        workflow.add_node("evaluator", self._evaluator_node)

        # Add edges
        workflow.add_edge("generator", "reflector")
        workflow.add_edge("reflector", "curator")
        workflow.add_edge("curator", "evaluator")
        workflow.add_edge("evaluator", END)

        # Set entry point
        workflow.set_entry_point("generator")

        # Compile graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _generator_node(self, state: ACEState) -> ACEState:
        """Generator node: produce outputs for all tasks."""
        try:
            logger.info(f"Generator node: processing {len(state['tasks'])} tasks")

            # Generate outputs for all tasks
            generated_outputs = self.generator.generate_batch(state["tasks"], state["playbook"])

            # Validate outputs
            validation_results = []
            for i, output in enumerate(generated_outputs):
                validation = self.generator.validate_response(output, state["tasks"][i])
                validation_results.append(validation)

            # Update state
            state["generated_outputs"] = generated_outputs

            logger.info(f"Generated {len(generated_outputs)} outputs")
            return state

        except Exception as e:
            logger.error(f"Generator node failed: {e}")
            state["error"] = f"Generator failed: {e!s}"
            return state

    def _reflector_node(self, state: ACEState) -> ACEState:
        """Reflector node: evaluate generated outputs."""
        try:
            logger.info("Reflector node: evaluating outputs")

            # Reflect on all generated outputs
            reflections = self.reflector.reflect_batch(
                state["tasks"], state["generated_outputs"], playbook=state["playbook"]
            )

            # Analyze reflection patterns
            reflection_analysis = self.reflector.analyze_reflection_patterns(reflections)

            # Update state
            state["reflections"] = reflections

            logger.info(f"Completed reflection on {len(reflections)} outputs")
            return state

        except Exception as e:
            logger.error(f"Reflector node failed: {e}")
            state["error"] = f"Reflector failed: {e!s}"
            return state

    def _curator_node(self, state: ACEState) -> ACEState:
        """Curator node: update playbook based on reflections."""
        try:
            logger.info("Curator node: updating playbook")

            # Update playbook based on reflections
            updated_playbook = self.curator.curate(state["playbook"], state["reflections"], state["iteration"])

            # Update state
            state["updated_playbook"] = updated_playbook

            logger.info(f"Playbook updated to version {updated_playbook.version}")
            return state

        except Exception as e:
            logger.error(f"Curator node failed: {e}")
            state["error"] = f"Curator failed: {e!s}"
            return state

    def _evaluator_node(self, state: ACEState) -> ACEState:
        """Evaluator node: calculate metrics for the iteration."""
        try:
            logger.info("Evaluator node: calculating metrics")

            # Calculate iteration metrics (reflection/output quality)
            metrics = self._calculate_iteration_metrics(state)

            # If evaluation dataset is available, run quantitative evaluation and persist into playbook
            if self.evaluation_dataset:
                logger.info("Running quantitative evaluation on evaluation dataset for gating")
                prev_pm = state["playbook"].metadata.performance_metrics

                eval_results = self.evaluator.evaluate_playbook(
                    state["updated_playbook"], self.evaluation_dataset, self.generator
                )
                q = eval_results.get("quantitative_metrics", {})
                task_acc = eval_results.get("task_accuracy", {})

                # Persist metrics into updated playbook
                self.evaluator.apply_metrics_to_playbook(
                    state["updated_playbook"],
                    q,
                    task_accuracy=task_acc,
                )

                # Compute deltas for gating
                new_bleu = float(q.get("bleu", 0.0))
                new_em = float(q.get("exact_match", 0.0))
                delta_bleu = new_bleu - float(prev_pm.bleu_score or 0.0)
                delta_em = new_em - float(prev_pm.exact_match or 0.0)

                # Add to iteration metrics
                metrics.update({
                    "bleu": new_bleu,
                    "exact_match": new_em,
                    "rougeL": float(q.get("rougeL", 0.0)),
                    "semantic_similarity": float(q.get("semantic_similarity", 0.0)),
                    "overall_accuracy": float(task_acc.get("overall_accuracy", 0.0)),
                    "delta_bleu": delta_bleu,
                    "delta_exact_match": delta_em,
                })

                # Set convergence status based on thresholds
                if delta_bleu >= self.bleu_threshold or delta_em >= self.em_threshold:
                    state["updated_playbook"].metadata.convergence_status = "improving"
                elif delta_bleu < 0 or delta_em < 0:
                    state["updated_playbook"].metadata.convergence_status = "degraded"
                else:
                    state["updated_playbook"].metadata.convergence_status = "plateaued"

            # Update state
            state["metrics"] = metrics

            logger.info("Metrics calculated successfully")
            return state

        except Exception as e:
            logger.error(f"Evaluator node failed: {e}")
            state["error"] = f"Evaluator failed: {e!s}"
            return state

    def _calculate_iteration_metrics(self, state: ACEState) -> Dict[str, Any]:
        """Calculate metrics for the current iteration."""
        metrics = {
            "iteration": state["iteration"],
            "num_tasks": len(state["tasks"]),
            "num_outputs": len(state["generated_outputs"]),
            "num_reflections": len(state["reflections"]),
            "playbook_version": state["updated_playbook"].version,
            "playbook_heuristics": len(state["updated_playbook"].context.heuristics),
            "playbook_examples": len(state["updated_playbook"].context.few_shot_examples),
        }

        # Calculate average reflection scores
        if state["reflections"]:
            scores = [r.get("overall_score", 0.0) for r in state["reflections"]]
            metrics["avg_reflection_score"] = sum(scores) / len(scores)
            metrics["min_reflection_score"] = min(scores)
            metrics["max_reflection_score"] = max(scores)

        # Calculate output quality metrics
        if state["generated_outputs"]:
            avg_length = sum(len(output) for output in state["generated_outputs"]) / len(state["generated_outputs"])
            code_examples = sum(1 for output in state["generated_outputs"] if "```" in output)
            citations = sum(
                1 for output in state["generated_outputs"] if any(url in output.lower() for url in ["http", "docs."])
            )

            metrics["avg_output_length"] = avg_length
            metrics["code_example_rate"] = code_examples / len(state["generated_outputs"])
            metrics["citation_rate"] = citations / len(state["generated_outputs"])

        return metrics

    def run_iteration(self, playbook: Playbook, tasks: List[str], iteration: int) -> Dict[str, Any]:
        """
        Run a single ACE iteration.

        Args:
            playbook: Current playbook
            tasks: List of tasks to process
            iteration: Current iteration number

        Returns:
            Results of the iteration
        """
        # Initialize state
        initial_state = ACEState(
            playbook=playbook,
            tasks=tasks,
            generated_outputs=[],
            reflections=[],
            updated_playbook=playbook,
            iteration=iteration,
            metrics={},
            error=None,
        )

        # Run the workflow
        try:
            result = self.graph.invoke(
                initial_state, config={"configurable": {"thread_id": f"ace_iteration_{iteration}"}}
            )

            if result.get("error"):
                logger.error(f"Iteration {iteration} failed: {result['error']}")
                return {"success": False, "error": result["error"], "iteration": iteration}

            return {
                "success": True,
                "iteration": iteration,
                "updated_playbook": result["updated_playbook"],
                "metrics": result["metrics"],
                "generated_outputs": result["generated_outputs"],
                "reflections": result["reflections"],
            }

        except Exception as e:
            logger.error(f"Iteration {iteration} failed with exception: {e}")
            return {"success": False, "error": str(e), "iteration": iteration}

    def run_iterations(
        self,
        playbook: Playbook,
        tasks: List[str],
        max_iterations: int = 10,
        evaluation_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple ACE iterations.

        Args:
            playbook: Initial playbook
            tasks: List of tasks to process
            max_iterations: Maximum number of iterations

        Returns:
            Results of all iterations
        """
        results = {"iterations": [], "final_playbook": playbook, "convergence_reached": False, "total_iterations": 0}

        current_playbook = playbook
        # If evaluation_dataset provided per call, override instance default
        if evaluation_dataset is not None:
            self.evaluation_dataset = evaluation_dataset

        for iteration in range(1, max_iterations + 1):
            logger.info(f"Starting iteration {iteration}/{max_iterations}")

            # Run iteration
            iteration_result = self.run_iteration(current_playbook, tasks, iteration)

            if not iteration_result["success"]:
                logger.error(f"Iteration {iteration} failed, stopping")
                break

            # Update playbook for next iteration
            current_playbook = iteration_result["updated_playbook"]

            # Store results
            results["iterations"].append(iteration_result)
            results["final_playbook"] = current_playbook
            results["total_iterations"] = iteration

            # Check for convergence
            if self._check_convergence(results["iterations"]):
                results["convergence_reached"] = True
                logger.info(f"Convergence reached at iteration {iteration}")
                break

        return results

    def _check_convergence(self, iterations: List[Dict[str, Any]], patience: int = 2) -> bool:
        """
        Check if the system has converged.

        Args:
            iterations: List of iteration results
            threshold: Improvement threshold for convergence
            patience: Number of iterations without improvement

        Returns:
            True if converged, False otherwise
        """
        if len(iterations) < patience + 1:
            return False

        # Use BLEU/EM deltas if available; otherwise, do not claim convergence
        recent = iterations[-patience - 1 :]
        deltas = []
        for it in recent:
            m = it.get("metrics", {})
            if "delta_bleu" in m or "delta_exact_match" in m:
                deltas.append((m.get("delta_bleu", 0.0), m.get("delta_exact_match", 0.0)))

        if len(deltas) < patience + 1:
            return False

        # Converged if across the recent window, no iteration clears thresholds
        no_improvements = 0
        for db, de in deltas:
            if db < self.bleu_threshold and de < self.em_threshold:
                no_improvements += 1

        return no_improvements >= patience + 1

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "nodes": ["generator", "reflector", "curator", "evaluator"],
            "edges": [
                ("generator", "reflector"),
                ("reflector", "curator"),
                ("curator", "evaluator"),
                ("evaluator", "END"),
            ],
            "state_schema": ACEState.__annotations__,
            "checkpointing_enabled": True,
        }
