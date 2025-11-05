"""
Tests for playbook schema and validation.
"""

import tempfile
from pathlib import Path

import pytest

from agentic_context_engineering.playbook_schema import (
    FewShotExample,
    Heuristic,
    PerformanceMetrics,
    Playbook,
    PlaybookContext,
    PlaybookMetadata,
)
from agentic_context_engineering.utils.versioning import compare_versions, increment_version


class TestPlaybookSchema:
    """Test playbook schema validation and operations."""

    def test_heuristic_creation(self):
        """Test heuristic creation and validation."""
        heuristic = Heuristic(
            id="h001", rule="Always provide code examples", confidence=0.85, created_iteration=1, last_updated=1
        )

        assert heuristic.id == "h001"
        assert heuristic.confidence == 0.85
        assert heuristic.usage_count == 0  # Default value
        assert heuristic.success_rate == 0.0  # Default value

    def test_heuristic_validation(self):
        """Test heuristic validation rules."""
        # Valid heuristic
        heuristic = Heuristic(id="h001", rule="Test rule", confidence=0.5, created_iteration=0, last_updated=0)
        assert heuristic.confidence == 0.5

        # Invalid confidence (should be clamped)
        with pytest.raises(ValueError):
            Heuristic(
                id="h001",
                rule="Test rule",
                confidence=1.5,  # Invalid: > 1.0
                created_iteration=0,
                last_updated=0,
            )

    def test_few_shot_example(self):
        """Test few-shot example creation."""
        example = FewShotExample(
            input="How to read JSON?", output="Use json.load()", annotation="Good example", quality_score=0.9
        )

        assert example.input == "How to read JSON?"
        assert example.quality_score == 0.9

    def test_performance_metrics(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(accuracy=0.85, bleu_score=0.75, avg_tokens=150, inference_time_sec=2.5)

        assert metrics.accuracy == 0.85
        assert metrics.bleu_score == 0.75
        assert metrics.avg_tokens == 150

    def test_playbook_creation(self):
        """Test complete playbook creation."""
        # Create components
        heuristic = Heuristic(id="h001", rule="Test rule", confidence=0.8, created_iteration=0, last_updated=0)

        example = FewShotExample(input="Test input", output="Test output", annotation="Test annotation")

        context = PlaybookContext(
            system_instructions="You are a helpful assistant.",
            heuristics=[heuristic],
            few_shot_examples=[example],
            constraints=["Be helpful"],
        )

        metadata = PlaybookMetadata(
            created_at="2025-01-27T10:00:00Z",
            iteration=0,
            performance_metrics=PerformanceMetrics(accuracy=0.8, avg_tokens=100, inference_time_sec=1.0),
            convergence_status="improving",
        )

        # Create playbook
        playbook = Playbook(version="1.0.0", metadata=metadata, context=context)

        assert playbook.version == "1.0.0"
        assert len(playbook.context.heuristics) == 1
        assert len(playbook.context.few_shot_examples) == 1

    def test_playbook_yaml_serialization(self):
        """Test playbook YAML serialization and deserialization."""
        # Create a simple playbook
        heuristic = Heuristic(id="h001", rule="Test rule", confidence=0.8, created_iteration=0, last_updated=0)

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

        # Test YAML serialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            playbook.to_yaml(temp_path)

            # Test YAML deserialization
            loaded_playbook = Playbook.from_yaml(temp_path)

            assert loaded_playbook.version == playbook.version
            assert len(loaded_playbook.context.heuristics) == len(playbook.context.heuristics)
            assert loaded_playbook.context.system_instructions == playbook.context.system_instructions

        finally:
            Path(temp_path).unlink()

    def test_playbook_heuristic_operations(self):
        """Test playbook heuristic operations."""
        # Create base playbook
        context = PlaybookContext(system_instructions="Test", heuristics=[], few_shot_examples=[], constraints=[])

        metadata = PlaybookMetadata(
            created_at="2025-01-27T10:00:00Z",
            iteration=0,
            performance_metrics=PerformanceMetrics(accuracy=0.8, avg_tokens=100, inference_time_sec=1.0),
            convergence_status="improving",
        )

        playbook = Playbook(version="1.0.0", metadata=metadata, context=context)

        # Add heuristic
        heuristic_id = playbook.add_heuristic("New rule", 0.9, 1)
        assert heuristic_id == "h001"
        assert len(playbook.context.heuristics) == 1

        # Update heuristic
        success = playbook.update_heuristic("h001", "Updated rule", 0.95, 2)
        assert success
        assert playbook.context.heuristics[0].rule == "Updated rule"
        assert playbook.context.heuristics[0].confidence == 0.95

        # Get heuristic by ID
        heuristic = playbook.get_heuristic_by_id("h001")
        assert heuristic is not None
        assert heuristic.rule == "Updated rule"

    def test_playbook_example_operations(self):
        """Test playbook example operations."""
        # Create base playbook
        context = PlaybookContext(system_instructions="Test", heuristics=[], few_shot_examples=[], constraints=[])

        metadata = PlaybookMetadata(
            created_at="2025-01-27T10:00:00Z",
            iteration=0,
            performance_metrics=PerformanceMetrics(accuracy=0.8, avg_tokens=100, inference_time_sec=1.0),
            convergence_status="improving",
        )

        playbook = Playbook(version="1.0.0", metadata=metadata, context=context)

        # Add example
        playbook.add_example("Test input", "Test output", "Test annotation", 0.9, 1)

        assert len(playbook.context.few_shot_examples) == 1
        assert playbook.context.few_shot_examples[0].input == "Test input"
        assert playbook.context.few_shot_examples[0].quality_score == 0.9


class TestVersioning:
    """Test versioning utilities."""

    def test_increment_version(self):
        """Test version increment functionality."""
        # Test patch increment
        assert increment_version("1.0.0", "patch") == "1.0.1"
        assert increment_version("1.2.3", "patch") == "1.2.4"

        # Test minor increment
        assert increment_version("1.0.0", "minor") == "1.1.0"
        assert increment_version("1.2.3", "minor") == "1.3.0"

        # Test major increment
        assert increment_version("1.0.0", "major") == "2.0.0"
        assert increment_version("1.2.3", "major") == "2.0.0"

    def test_compare_versions(self):
        """Test version comparison."""
        # Test equal versions
        assert compare_versions("1.0.0", "1.0.0") == 0

        # Test version 1 < version 2
        assert compare_versions("1.0.0", "1.0.1") == -1
        assert compare_versions("1.0.0", "1.1.0") == -1
        assert compare_versions("1.0.0", "2.0.0") == -1

        # Test version 1 > version 2
        assert compare_versions("1.0.1", "1.0.0") == 1
        assert compare_versions("1.1.0", "1.0.0") == 1
        assert compare_versions("2.0.0", "1.0.0") == 1

    def test_invalid_version_format(self):
        """Test handling of invalid version formats."""
        with pytest.raises(ValueError):
            increment_version("1.0", "patch")  # Missing patch version

        with pytest.raises(ValueError):
            increment_version("1.0.0.0", "patch")  # Too many parts

        with pytest.raises(ValueError):
            increment_version("1.0.0", "invalid")  # Invalid bump type
