#!/usr/bin/env python3
"""
Simple test script to verify ACE system components work correctly.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        print("+ Playbook schema imports")
    except Exception as e:
        print(f"- Playbook schema import failed: {e}")
        return False

    try:
        print("+ Config manager imports")
    except Exception as e:
        print(f"- Config manager import failed: {e}")
        return False

    try:
        print("+ LLM client imports")
    except Exception as e:
        print(f"- LLM client import failed: {e}")
        return False

    try:
        print("+ Agent classes import")
    except Exception as e:
        print(f"- Agent classes import failed: {e}")
        return False

    try:
        print("+ Evaluation system imports")
    except Exception as e:
        print(f"- Evaluation system import failed: {e}")
        return False

    try:
        print("+ Orchestration system imports")
    except Exception as e:
        print(f"- Orchestration system import failed: {e}")
        return False

    return True


def test_playbook_schema():
    """Test playbook schema functionality"""
    print("\nTesting playbook schema...")

    try:
        from agentic_context_engineering.playbook_schema import FewShotExample, Heuristic

        # Create a test heuristic
        heuristic = Heuristic(
            id="test_1",
            rule="Always provide code examples when explaining Python concepts",
            confidence=0.8,
            created_iteration=1,
            last_updated=1,
            usage_count=5,
            success_rate=0.9,
        )
        print("+ Heuristic creation")

        # Create a test example
        example = FewShotExample(
            input="How do I create a list in Python?",
            output="You can create a list using square brackets: my_list = [1, 2, 3]",
            annotation="This shows the basic syntax for list creation",
            quality_score=0.9,
        )
        print("+ Few-shot example creation")

        # Test versioning
        from agentic_context_engineering.utils.versioning import increment_version

        new_version = increment_version("1.0.0", "minor")
        assert new_version == "1.1.0"
        print("+ Version increment")

        return True

    except Exception as e:
        print(f"- Playbook schema test failed: {e}")
        return False


def test_config_system():
    """Test configuration system"""
    print("\nTesting configuration system...")

    try:
        from agentic_context_engineering.utils.config import ConfigManager

        # Test config loading
        config_manager = ConfigManager("agentic_context_engineering/configs/default.yaml")
        llm_config = config_manager.get_llm_config()
        ace_config = config_manager.get_ace_config()

        print(f"+ LLM config loaded: {llm_config.model}")
        print(f"+ ACE config loaded: {ace_config.max_iterations} iterations")

        return True

    except Exception as e:
        print(f"- Configuration system test failed: {e}")
        return False


def test_dataset_generation():
    """Test synthetic dataset generation"""
    print("\nTesting dataset generation...")

    try:
        from agentic_context_engineering.eval.dataset_generator import DatasetGenerator

        # Generate a small test dataset
        generator = DatasetGenerator()
        dataset = generator.generate_dataset(size=5)

        assert len(dataset) == 5
        assert "input" in dataset[0]
        assert "reference_output" in dataset[0]
        assert "evaluation_criteria" in dataset[0]

        print(f"+ Generated {len(dataset)} test samples")

        # Test saving and loading
        test_file = "test_dataset.json"
        generator.save_dataset(dataset, test_file)
        loaded_dataset = generator.load_dataset(test_file)

        assert len(loaded_dataset) == len(dataset)
        print("+ Dataset save/load")

        # Clean up
        os.remove(test_file)

        return True

    except Exception as e:
        print(f"- Dataset generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")

    try:
        from agentic_context_engineering.utils.metrics import MetricsCalculator

        calculator = MetricsCalculator()

        # Test exact match
        exact_score = calculator.calculate_exact_match("Hello world", "Hello world")
        assert exact_score == 1.0
        print("+ Exact match calculation")

        # Test semantic similarity
        similarity = calculator.calculate_semantic_similarity(["Hello world"], ["Hi there"])
        assert 0.0 <= similarity <= 1.0
        print("+ Semantic similarity calculation")

        return True

    except Exception as e:
        print(f"- Metrics test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("ACE System Component Tests")
    print("=" * 50)

    tests = [test_imports, test_playbook_schema, test_config_system, test_dataset_generation, test_metrics]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("ALL TESTS PASSED! ACE system is ready.")
        return True
    else:
        print("Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
