<!-- Integration guide for external projects using ACE -->
# Integrating Agentic Context Engineering (ACE) Into External Projects

This guide explains how to consume the **Agentic Context Engineering (ACE)** framework from another codebase—such as the IBM Gen AI Companion project—and how to operate the ACE loop on your own domain conversations.

---

## 1. Installation Options

You can install ACE directly from GitHub or as a local editable dependency.

### 1.1 Install from GitHub

```bash
pip install "agentic-context-engineering @ git+https://github.com/EgeCankaya/Agentic-Context-Engineering.git"
```

### 1.2 Install from a Local Checkout

```bash
git clone https://github.com/EgeCankaya/Agentic-Context-Engineering.git
cd Agentic-Context-Engineering
pip install -e .
```

**Python requirement:** 3.9 or higher.

---

## 2. Core Imports

Once installed, import ACE components from the package root:

```python
from agentic_context_engineering import (
    ACERunner,
    Playbook,
    LLMClient,
    ConfigManager,
)
```

Additional utilities added for external integrations:

```python
from agentic_context_engineering import ConversationLogger, EducationalEvaluator
```

---

## 3. Loading and Using Playbook Context

The ACE playbook stores system instructions, heuristics, few-shot examples, and metadata. Load the latest playbook and extract the pieces you need for prompts or RAG pipelines.

```python
from agentic_context_engineering import Playbook

playbook = Playbook.from_yaml("outputs/playbook_v1.2.0.yaml")

system_prompt = playbook.get_system_prompt()
context_bundle = playbook.export_context_for_rag()
examples = playbook.get_relevant_examples(task={"course": "Course 5", "topic": "Flask"})
```

**Recommended workflow:**

1. Start with the base playbook (`agentic_context_engineering/playbook_schema/base_playbook.yaml`).
2. After running ACE, use the latest playbook from `outputs/`.
3. Call `playbook.export_context_for_rag()` to receive structured content ready for prompt assembly.

---

## 4. Using ACE With a RAG Pipeline

You can combine ACE guidance with your own retrieval results using the `Generator` agent.

```python
from agentic_context_engineering.agents import Generator
from agentic_context_engineering.utils import LLMClient

llm_client = LLMClient()
generator = Generator(llm_client)

retrieved_docs = [
    {
        "id": "course12_transformers",
        "content": "Transformers rely on multi-head self-attention...",
        "metadata": {"course": 12, "module": "Attention"},
    },
    # ... more retrieved chunks
]

response = generator.generate_with_context(
    task="Explain how multi-head attention works in transformers.",
    playbook=playbook,
    retrieved_docs=retrieved_docs,
)

citations = generator.extract_citations(response, retrieved_docs)
```

**Key steps:**

- `construct_rag_prompt()` combines system instructions, heuristics, examples, and retrieved chunks.
- `generate_with_context()` invokes the LLM using the assembled prompt.
- `extract_citations()` maps inline references back to retrieved sources.

---

## 5. Logging Conversations for ACE

Project 2 will log multi-turn conversations and feed them into ACE. Use the provided `ConversationLogger` utility.

```python
from agentic_context_engineering import ConversationLogger

logger = ConversationLogger(output_dir="outputs/conversations")

session_id = logger.start_session(user_id="student_001", metadata={"mode": "study"})

logger.log_turn(
    session_id=session_id,
    question="How do I fine-tune a transformer on custom text?",
    answer=response,
    retrieved_docs=retrieved_docs,
    annotations={
        "course": 13,
        "confidence": 0.82,
        "user_feedback": "helpful",
    },
)

logger.end_session(session_id)

dataset = logger.export_for_ace()
```

The exported dataset aligns with ACE's expectation: each item contains `input`, `reference_output`, `evaluation_criteria`, and optional metadata. Feed this into ACE as `tasks` or evaluation sets.

---

## 6. Running ACE on Custom Data

Use `ACERunner` to execute Generator → Reflector → Curator cycles on your domain-specific tasks.

```python
from agentic_context_engineering import ACERunner

runner = ACERunner(config_path="agentic_context_engineering/configs/default.yaml")

custom_tasks = [item["input"] for item in dataset]

results = runner.run_iterations(
    num_iterations=5,
    tasks=custom_tasks,
    playbook_path="outputs/ace_playbooks/playbook_v1.2.0.yaml",
)

print(results["final_playbook"].version)
```

To gate ACE with evaluation metrics, provide an evaluation dataset:

```python
results = runner.run_iterations(
    num_iterations=5,
    tasks=custom_tasks,
    evaluation_dataset=dataset,  # must contain input/reference_output pairs
)
```

The runner saves updated playbooks under `outputs/` and persists metrics history for comparison.

---

## 7. Evaluating Educational Conversations

For learning-focused agents, combine ACE's evaluator with the `EducationalEvaluator` for rubric-based scoring.

```python
from agentic_context_engineering import EducationalEvaluator

edu_eval = EducationalEvaluator()

scores = edu_eval.evaluate_conversation(
    question="Explain backpropagation in simple terms",
    answer=response,
    retrieved_docs=retrieved_docs,
)

print(scores)
```

Metrics include accuracy, grounding, pedagogy, citation quality, completeness, clarity, and appropriate use of "I don't know".

---

## 8. Recommended Workflow for External Projects

1. **Ingest Domain Content** → Build your vector store.
2. **Load ACE Playbook** → `Playbook.from_yaml(...)`.
3. **Answer Questions** → Use `Generator.generate_with_context()` with retrieved docs.
4. **Log Conversations** → `ConversationLogger` records Q&A turns.
5. **Trigger ACE Cycles** → Run `ACERunner.run_iterations()` on recent conversations.
6. **Reload Playbook** → Update your app with the new playbook version.
7. **Evaluate Progress** → Compare baseline vs. ACE-enhanced using `EducationalEvaluator` and ACE metrics.

---

## 9. File & Directory Expectations

| Resource | Location | Purpose |
|----------|----------|---------|
| Base playbook | `agentic_context_engineering/playbook_schema/base_playbook.yaml` | Starting context |
| Updated playbooks | `outputs/playbook_v*.yaml` | Generated by ACE runs |
| Conversation logs | `outputs/conversations/*.json` | Generated by `ConversationLogger` |
| Evaluation results | `outputs/evaluation_*.json` | Produced by `ACERunner.evaluate_playbook()` |

---

## 10. Troubleshooting Checklist

- **Ollama not reachable:** Ensure `ollama serve` is running and the model is pulled.
- **CUDA errors:** Switch to a smaller model (e.g., `mistral:7b-instruct`) or use CPU-only mode via Ollama.
- **Missing fields in exported dataset:** Confirm you call `logger.end_session()` and `logger.export_for_ace()`.
- **Playbook not updating:** Check that ACE iterations meet improvement thresholds (`gate_bleu_delta`, `gate_em_delta`).
- **High latency:** Enable `warm_start` in config and cache retrieved docs to minimize LLM calls.

---

## 11. Next Steps for Project 2

1. Set up the IBM Gen AI Companion repository.
2. Ingest IBM course materials into a vector store.
3. Use this guide to hook ACE into the companion chatbot.
4. Run baseline (without ACE) vs. ACE-enhanced comparisons.
5. Share updated playbooks and evaluation metrics with your team.

For further reference, explore the example project under `examples/external_rag_integration/` once added.
