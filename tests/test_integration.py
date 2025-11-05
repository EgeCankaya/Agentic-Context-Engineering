"""
Integration tests for ACE system.
Tests the complete workflow from initialization to iteration execution.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from agentic_context_engineering import ACERunner
from agentic_context_engineering.utils.config import ConfigManager


class TestACEIntegration:
    """Integration tests for the complete ACE system."""

    @patch("agentic_context_engineering.runners.ace_runner.LLMClient")
    @patch("agentic_context_engineering.runners.ace_runner.ACEOrchestrator")
    def test_ace_runner_initialization(self, mock_orchestrator, mock_llm_client):
        """Test ACE runner initialization."""
        # Mock LLM client
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance

        # Mock orchestrator
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            f.write("""
llm:
  provider: "ollama"
  model: "llama3.1:8b-instruct-fp16"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 2000

ace:
  max_iterations: 5
  convergence_threshold: 0.05
  reflection_batch_size: 3
  early_stopping_patience: 2

evaluation:
  metrics: ["bleu", "rouge", "exact_match"]
  holdout_ratio: 0.2

performance:
  max_concurrent_requests: 1
  warm_start: true
  log_vram_usage: true

logging:
  level: "INFO"
  save_intermediate: true
  output_dir: "./test_outputs"
  structured_format: "json"
""")

        try:
            # Initialize runner
            runner = ACERunner(config_path=config_path)

            # Verify initialization
            assert runner.config is not None
            assert runner.llm_client == mock_llm_instance
            assert runner.orchestrator == mock_orchestrator_instance

        finally:
            Path(config_path).unlink()

    @patch("agentic_context_engineering.runners.ace_runner.LLMClient")
    @patch("agentic_context_engineering.runners.ace_runner.ACEOrchestrator")
    def test_ace_runner_iterations(self, mock_orchestrator, mock_llm_client):
        """Test ACE runner iteration execution."""
        # Mock LLM client
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance

        # Mock orchestrator with successful iteration
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.run_iterations.return_value = {
            "iterations": [
                {"success": True, "iteration": 1, "updated_playbook": Mock(), "metrics": {"avg_reflection_score": 0.8}}
            ],
            "final_playbook": Mock(),
            "convergence_reached": False,
            "total_iterations": 1,
        }
        mock_orchestrator.return_value = mock_orchestrator_instance

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            f.write("""
llm:
  provider: "ollama"
  model: "llama3.1:8b-instruct-fp16"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 2000

ace:
  max_iterations: 5
  convergence_threshold: 0.05
  reflection_batch_size: 3
  early_stopping_patience: 2

evaluation:
  metrics: ["bleu", "rouge", "exact_match"]
  holdout_ratio: 0.2

performance:
  max_concurrent_requests: 1
  warm_start: true
  log_vram_usage: true

logging:
  level: "INFO"
  save_intermediate: true
  output_dir: "./test_outputs"
  structured_format: "json"
""")

        try:
            # Initialize runner
            runner = ACERunner(config_path=config_path)

            # Run iterations
            results = runner.run_iterations(num_iterations=1)

            # Verify results
            assert results["total_iterations"] == 1
            assert results["convergence_reached"] == False
            assert "runtime_seconds" in results
            assert "configuration" in results

            # Verify orchestrator was called
            mock_orchestrator_instance.run_iterations.assert_called_once()

        finally:
            Path(config_path).unlink()

    def test_config_manager_loading(self):
        """Test configuration manager functionality."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            f.write("""
llm:
  provider: "ollama"
  model: "test-model"
  base_url: "http://localhost:11434"
  temperature: 0.5
  max_tokens: 1000

ace:
  max_iterations: 3
  convergence_threshold: 0.1
  reflection_batch_size: 2
  early_stopping_patience: 1

evaluation:
  metrics: ["bleu", "rouge"]
  holdout_ratio: 0.3

performance:
  max_concurrent_requests: 2
  warm_start: false
  log_vram_usage: false

logging:
  level: "DEBUG"
  save_intermediate: false
  output_dir: "./custom_outputs"
  structured_format: "yaml"
""")

        try:
            # Test config loading
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()

            # Verify config values
            assert config["llm"]["model"] == "test-model"
            assert config["llm"]["temperature"] == 0.5
            assert config["ace"]["max_iterations"] == 3
            assert config["logging"]["level"] == "DEBUG"

            # Test Pydantic model conversion
            llm_config = config_manager.get_llm_config()
            assert llm_config.model == "test-model"
            assert llm_config.temperature == 0.5

        finally:
            Path(config_path).unlink()

    def test_playbook_workflow(self):
        """Test complete playbook workflow."""
        # Create a simple playbook
        from agentic_context_engineering.playbook_schema import (
            Heuristic,
            PerformanceMetrics,
            Playbook,
            PlaybookContext,
            PlaybookMetadata,
        )

        # Create components
        heuristic = Heuristic(id="h001", rule="Test heuristic", confidence=0.8, created_iteration=0, last_updated=0)

        context = PlaybookContext(
            system_instructions="Test instructions", heuristics=[heuristic], few_shot_examples=[], constraints=[]
        )

        metadata = PlaybookMetadata(
            created_at="2025-01-27T10:00:00Z",
            iteration=0,
            performance_metrics=PerformanceMetrics(accuracy=0.8, avg_tokens=100, inference_time_sec=1.0),
            convergence_status="improving",
        )

        playbook = Playbook(version="1.0.0", metadata=metadata, context=context)

        # Test playbook operations
        assert playbook.version == "1.0.0"
        assert len(playbook.context.heuristics) == 1

        # Test heuristic addition
        new_heuristic_id = playbook.add_heuristic("New rule", 0.9, 1)
        assert new_heuristic_id == "h002"
        assert len(playbook.context.heuristics) == 2

        # Test heuristic update
        success = playbook.update_heuristic("h001", "Updated rule", 0.95, 1)
        assert success
        assert playbook.context.heuristics[0].rule == "Updated rule"

        # Test example addition
        playbook.add_example("Test input", "Test output", "Test annotation", 0.9, 1)
        assert len(playbook.context.few_shot_examples) == 1

    @patch("agentic_context_engineering.runners.ace_runner.LLMClient")
    def test_system_status(self, mock_llm_client):
        """Test system status reporting."""
        # Mock LLM client
        mock_llm_instance = Mock()
        mock_llm_instance.health_check.return_value = True
        mock_llm_instance.get_vram_usage.return_value = {"allocated_gb": 8.5, "total_gb": 16.0, "utilization_pct": 53.1}
        mock_llm_client.return_value = mock_llm_instance

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            f.write("""
llm:
  provider: "ollama"
  model: "llama3.1:8b-instruct-fp16"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 2000

ace:
  max_iterations: 5
  convergence_threshold: 0.05
  reflection_batch_size: 3
  early_stopping_patience: 2

evaluation:
  metrics: ["bleu", "rouge", "exact_match"]
  holdout_ratio: 0.2

performance:
  max_concurrent_requests: 1
  warm_start: true
  log_vram_usage: true

logging:
  level: "INFO"
  save_intermediate: true
  output_dir: "./test_outputs"
  structured_format: "json"
""")

        try:
            # Initialize runner
            runner = ACERunner(config_path=config_path)

            # Get system status
            status = runner.get_system_status()

            # Verify status structure
            assert "llm_client" in status
            assert "orchestrator" in status
            assert "output_directory" in status
            assert "configuration" in status

            # Verify LLM client status
            llm_status = status["llm_client"]
            assert llm_status["model"] == "llama3.1:8b-instruct-fp16"
            assert llm_status["health_check"] == True
            assert "vram_usage" in llm_status

        finally:
            Path(config_path).unlink()
