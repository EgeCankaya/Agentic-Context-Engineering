"""
Synthetic dataset generator for ACE evaluation.
Creates technical Q&A datasets for Python development tasks.
"""

import json
import random
from typing import Any, Dict, List, Tuple


class DatasetGenerator:
    """Generates synthetic technical Q&A datasets for ACE evaluation."""

    def __init__(self, seed: int = 42):
        """
        Initialize dataset generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.seed = seed

        # Template questions and answers
        self.question_templates = [
            "How do I {action} in Python?",
            "What's the best way to {action} in Python?",
            "How can I {action} using Python?",
            "What's the proper way to {action} in Python?",
            "How do I implement {concept} in Python?",
            "What's the recommended approach for {concept} in Python?",
            "How can I handle {scenario} in Python?",
            "What's the most efficient way to {action} in Python?",
        ]

        self.actions = [
            "read a JSON file",
            "write to a CSV file",
            "make HTTP requests",
            "handle exceptions",
            "parse command line arguments",
            "work with dates and times",
            "create a virtual environment",
            "install packages",
            "debug code",
            "profile performance",
            "handle file I/O",
            "work with regular expressions",
            "create a REST API",
            "connect to a database",
            "handle async operations",
            "implement caching",
            "handle authentication",
            "work with environment variables",
            "create unit tests",
            "handle logging",
        ]

        self.concepts = [
            "error handling with try-except",
            "list comprehensions",
            "decorators",
            "context managers",
            "generators",
            "lambda functions",
            "class inheritance",
            "method overloading",
            "property decorators",
            "static methods",
            "abstract base classes",
            "metaclasses",
            "descriptors",
            "closures",
            "recursion",
        ]

        self.scenarios = [
            "file not found errors",
            "network timeouts",
            "memory issues",
            "concurrent access",
            "data validation",
            "API rate limiting",
            "database connection failures",
            "large file processing",
            "user input validation",
            "cross-platform compatibility",
        ]

    def generate_dataset(self, size: int = 50) -> List[Dict[str, Any]]:
        """
        Generate a synthetic dataset of technical Q&A pairs.

        Args:
            size: Number of samples to generate

        Returns:
            List of dataset samples
        """
        dataset = []

        for i in range(size):
            sample = self._generate_sample(i)
            dataset.append(sample)

        return dataset

    def _generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a single dataset sample."""
        # Choose question type
        question_type = random.choice(["action", "concept", "scenario"])

        if question_type == "action":
            action = random.choice(self.actions)
            # Filter templates that support {action}
            action_templates = [t for t in self.question_templates if "{action}" in t]
            question = random.choice(action_templates).format(action=action)
            answer = self._generate_action_answer(action)
        elif question_type == "concept":
            concept = random.choice(self.concepts)
            # Filter templates that support {concept}
            concept_templates = [t for t in self.question_templates if "{concept}" in t]
            question = random.choice(concept_templates).format(concept=concept)
            answer = self._generate_concept_answer(concept)
        else:  # scenario
            scenario = random.choice(self.scenarios)
            # Filter templates that support {scenario}
            scenario_templates = [t for t in self.question_templates if "{scenario}" in t]
            question = random.choice(scenario_templates).format(scenario=scenario)
            answer = self._generate_scenario_answer(scenario)

        # Generate evaluation criteria
        criteria = self._generate_evaluation_criteria(question_type)

        # Determine difficulty
        difficulty = self._determine_difficulty(question, answer)

        # Generate tags
        tags = self._generate_tags(question, question_type)

        return {
            "id": f"task_{index + 1:03d}",
            "input": question,
            "reference_output": answer,
            "evaluation_criteria": criteria,
            "difficulty": difficulty,
            "tags": tags,
            "question_type": question_type,
        }

    def _generate_action_answer(self, action: str) -> str:
        """Generate answer for action-based questions."""
        answers = {
            "read a JSON file": """Use the `json` module from the standard library:

```python
import json

# Read from file
with open('data.json', 'r') as f:
    data = json.load(f)

# Read from string
json_string = '{"key": "value"}'
data = json.loads(json_string)
```

**Documentation:** https://docs.python.org/3/library/json.html""",
            "write to a CSV file": """Use the `csv` module for structured CSV writing:

```python
import csv

data = [['Name', 'Age'], ['Alice', 30], ['Bob', 25]]

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
```

**Documentation:** https://docs.python.org/3/library/csv.html""",
            "make HTTP requests": """Use the `requests` library:

```python
import requests

# GET request
response = requests.get('https://api.example.com/data')
response.raise_for_status()  # Raise exception for bad status
data = response.json()

# POST request
response = requests.post('https://api.example.com/data',
                       json={'key': 'value'})
```

**Installation:** `pip install requests`
**Documentation:** https://requests.readthedocs.io/""",
            "handle exceptions": """Use try-except blocks with specific exception types:

```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    print("Operation successful")
finally:
    print("Cleanup code here")
```

**Documentation:** https://docs.python.org/3/tutorial/errors.html""",
        }

        return answers.get(action, f"To {action}, you can use Python's built-in modules and libraries.")

    def _generate_concept_answer(self, concept: str) -> str:
        """Generate answer for concept-based questions."""
        answers = {
            "error handling with try-except": """Use try-except blocks for error handling:

```python
try:
    # Risky operation
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Best practices:**
- Catch specific exceptions first
- Use `raise` to re-raise exceptions
- Use `finally` for cleanup code

**Documentation:** https://docs.python.org/3/tutorial/errors.html""",
            "list comprehensions": """List comprehensions provide a concise way to create lists:

```python
# Basic syntax: [expression for item in iterable if condition]

# Example: squares of even numbers
squares = [x**2 for x in range(10) if x % 2 == 0]

# Equivalent to:
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)
```

**Documentation:** https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions""",
            "decorators": """Decorators modify function behavior:

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"
```

**Documentation:** https://docs.python.org/3/glossary.html#term-decorator""",
        }

        return answers.get(concept, f"Here's how to implement {concept} in Python.")

    def _generate_scenario_answer(self, scenario: str) -> str:
        """Generate answer for scenario-based questions."""
        answers = {
            "file not found errors": """Handle FileNotFoundError with proper error handling:

```python
import os

try:
    with open('file.txt', 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("File not found. Creating default content.")
    with open('file.txt', 'w') as f:
        f.write("Default content")
except PermissionError:
    print("Permission denied")
```

**Documentation:** https://docs.python.org/3/library/exceptions.html#FileNotFoundError""",
            "network timeouts": """Handle network timeouts with requests:

```python
import requests
from requests.exceptions import Timeout

try:
    response = requests.get('https://api.example.com', timeout=5)
    response.raise_for_status()
except Timeout:
    print("Request timed out")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

**Documentation:** https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts""",
        }

        return answers.get(scenario, f"Here's how to handle {scenario} in Python.")

    def _generate_evaluation_criteria(self, question_type: str) -> Dict[str, str]:
        """Generate evaluation criteria for a question."""
        criteria = {
            "accuracy": "Response contains correct information",
            "completeness": "Response fully addresses the question",
            "clarity": "Response is well-structured and easy to understand",
        }

        if question_type == "action":
            criteria["code_quality"] = "Includes runnable code examples"
            criteria["documentation"] = "References official documentation"
        elif question_type == "concept":
            criteria["explanation"] = "Provides clear conceptual explanation"
            criteria["examples"] = "Includes practical examples"
        else:  # scenario
            criteria["error_handling"] = "Demonstrates proper error handling"
            criteria["robustness"] = "Shows robust solution approach"

        return criteria

    def _determine_difficulty(self, question: str, answer: str) -> str:
        """Determine difficulty level based on question and answer complexity."""
        if len(answer) < 200:
            return "easy"
        elif len(answer) < 500:
            return "medium"
        else:
            return "hard"

    def _generate_tags(self, question: str, question_type: str) -> List[str]:
        """Generate tags for the question."""
        tags = [question_type]

        # Add topic-based tags
        if "json" in question.lower():
            tags.append("json")
        elif "csv" in question.lower():
            tags.append("csv")
        elif "http" in question.lower() or "request" in question.lower():
            tags.append("networking")
        elif "error" in question.lower() or "exception" in question.lower():
            tags.append("error-handling")
        elif "file" in question.lower():
            tags.append("file-io")
        elif "test" in question.lower():
            tags.append("testing")
        elif "async" in question.lower():
            tags.append("asyncio")
        elif "database" in question.lower() or "db" in question.lower():
            tags.append("database")

        return tags

    def split_dataset(
        self,
        dataset: List[Dict[str, Any]],
        dev_ratio: float = 0.4,
        iteration_ratio: float = 0.4,
        holdout_ratio: float = 0.2,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into development, iteration, and holdout sets.

        Args:
            dataset: Full dataset
            dev_ratio: Ratio for development set
            iteration_ratio: Ratio for iteration set
            holdout_ratio: Ratio for holdout set

        Returns:
            Tuple of (dev_set, iteration_set, holdout_set)
        """
        # Shuffle dataset
        shuffled = dataset.copy()
        random.shuffle(shuffled)

        total_size = len(shuffled)
        dev_size = int(total_size * dev_ratio)
        iteration_size = int(total_size * iteration_ratio)

        dev_set = shuffled[:dev_size]
        iteration_set = shuffled[dev_size : dev_size + iteration_size]
        holdout_set = shuffled[dev_size + iteration_size :]

        return dev_set, iteration_set, holdout_set

    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str) -> None:
        """Save dataset to JSON file."""
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

    def load_dataset(self, input_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(input_path) as f:
            return json.load(f)


# Example usage
if __name__ == "__main__":
    generator = DatasetGenerator()

    # Generate full dataset
    dataset = generator.generate_dataset(size=50)

    # Split into sets
    dev_set, iteration_set, holdout_set = generator.split_dataset(dataset)

    print(f"Generated {len(dataset)} samples")
    print(f"Dev set: {len(dev_set)} samples")
    print(f"Iteration set: {len(iteration_set)} samples")
    print(f"Holdout set: {len(holdout_set)} samples")

    # Save datasets
    generator.save_dataset(dev_set, "eval/dev_set.json")
    generator.save_dataset(iteration_set, "eval/iteration_set.json")
    generator.save_dataset(holdout_set, "eval/holdout_set.json")
    generator.save_dataset(dataset, "eval/full_dataset.json")
