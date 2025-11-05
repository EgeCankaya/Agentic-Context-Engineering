# Product Requirements Document (PRD)
## Project 1: Agentic Context Engineering (ACE) — Evolving Contexts for Self-Improving Language Models

Version: 2.1 (adapted to current implementation)
Last Updated: October 29, 2025
Project Duration: Week 1 (10 days recommended)
Cost: $0 (100% local inference)

---

## 1. Overview

### Project Summary

This project implements a minimal version of Agentic Context Engineering (ACE) — a framework that enables self-improvement of large language models (LLMs) through context evolution rather than model fine-tuning. The system iteratively improves performance by running a three-phase loop:

1) Generator: Produces model outputs based on current context (playbook)
2) Reflector: Evaluates generated outputs, identifies weaknesses, and proposes improvements
3) Curator: Updates the structured playbook context based on reflection feedback

The system demonstrates that performance gains can be achieved through contextual optimization (prompt refinement, heuristic updates, evolving instruction sets) instead of adjusting model weights.

### Research Foundation

- Paper: `https://arxiv.org/abs/2510.04618`
- Key Idea: Context as a structured, evolving asset (playbook) that improves via generation→reflection→curation instead of weight updates

### Prerequisites

- Hardware: NVIDIA RTX 4070 Ti Super (16 GB VRAM) recommended; CPU-only supported but slower
- Skills: Python, prompt engineering, YAML/JSON
- Frameworks: LangGraph, LangChain
- Software: Ollama installed, at least one local model pulled

Note: Current default model is Mistral 7B Instruct via Ollama. Llama 3.1 8B can be used, but see Model Notes below.

---

## 2. Goals and Objectives

### Primary Goal

Develop a working ACE prototype that can:
- Maintain a structured, versioned playbook of contextual rules
- Run automatic generation → reflection → curation cycles
- Show improvement in responses over iterations
- Provide CLI interfaces for execution and evaluation

### Secondary Objectives

- Parse the ACE methodology into a reproducible architecture
- Design and validate a robust playbook schema (Pydantic)
- Implement agents (Generator, Reflector, Curator) orchestrated by LangGraph
- Provide a CLI for iterations, evaluation, and health checks
- Deliver documentation mapping paper concepts to implementation
- Achieve zero-cost operation using local inference

---

## 3. Key Features (Implemented)

- Playbook Schema: YAML + Pydantic models (`agentic_context_engineering/playbook_schema/schema.py`) with versioning and history
- Generator Agent: Builds prompts from playbook and generates responses
- Reflector Agent: Scores outputs and proposes improvements
- Curator Agent: Applies improvements to playbook with version bumping
- Orchestration: LangGraph `StateGraph` runs Gen → Ref → Cur → Eval
- Evaluation: BLEU/ROUGE/EM/Semantic similarity (sentence-transformers)
- CLI: Run iterations, evaluate, export, diff, check-gpu
- Outputs: Results and evolved playbooks saved under `outputs/`

Notes on current implementation differences:
- Default LLM Model: `mistral:7b-instruct` (Ollama). Llama 3.1 8B is supported but may require additional setup on Windows.
- Windows Unicode: Console output avoids emojis and some special glyphs to prevent `UnicodeEncodeError`.
- LangGraph: Checkpointer works via in-memory `MemorySaver`; per-iteration `thread_id` is provided at `graph.invoke` call.

---

## 3.5 Evaluation Dataset Specification

- Size: 30–50 tasks (dev/iteration/held-out splits supported)
- Type: Technical Q&A (Python-focused)
- Format: JSON objects with `input`, `reference_output`, `evaluation_criteria`, `difficulty`, `tags`
- Generator: `agentic_context_engineering/eval/dataset_generator.py`

Example item:
```json
{
  "id": "task_001",
  "input": "How do I implement retry logic with exponential backoff in Python?",
  "reference_output": "Use the tenacity library with @retry decorator...",
  "evaluation_criteria": {
    "accuracy": "Contains correct tenacity usage",
    "completeness": "Includes code example and explanation",
    "clarity": "Easy to understand for intermediate developers"
  },
  "difficulty": "medium",
  "tags": ["error-handling", "python", "libraries"]
}
```

---

## 4. Deliverables (As Implemented)

### Code Layout

```
agentic_context_engineering/
├── agents/
│   ├── generator.py
│   ├── reflector.py
│   └── curator.py
├── playbook_schema/
│   ├── schema.py
│   └── base_playbook.yaml
├── runners/
│   ├── ace_runner.py
│   └── orchestrator.py
├── utils/
│   ├── llm_client.py
│   ├── metrics.py
│   ├── config.py
│   └── versioning.py
├── eval/
│   ├── dataset_generator.py
│   └── evaluator.py
├── configs/
│   └── default.yaml
├── cli.py
└── outputs/ (generated)
```

### Command-Line Interface

```
# Run ACE iterations
python -m agentic_context_engineering.cli run --iterations 5 --tasks iteration_set.json

# Evaluate a playbook on a dataset
python -m agentic_context_engineering.cli evaluate --playbook agentic_context_engineering/playbook_schema/base_playbook.yaml --dataset test_dataset.json --output evaluation_results.json

# Compare two playbook YAMLs
python -m agentic_context_engineering.cli diff --playbook1 outputs/playbook_v1.0.0.yaml --playbook2 outputs/playbook_v1.0.1.yaml

# Export results (CSV/JSON)
python -m agentic_context_engineering.cli export --format json --output results.json --results-dir outputs

# System health (GPU + Ollama + generation test)
python -m agentic_context_engineering.cli check-gpu
```

---

## 5. Success Criteria

- Full Gen→Ref→Cur cycles complete without manual intervention
- Playbook updates persist between iterations and versions bump (e.g., 1.0.0 → 1.0.1)
- Metrics computed per iteration; qualitative improvements visible (more code examples, citations)
- Works locally using Ollama; GPU acceleration recommended but CPU mode supported

---

## 6. Technical Stack (Current)

- Language: Python 3.9+
- Frameworks: LangGraph, LangChain
- LLM Backend: Ollama
- Default Model: `mistral:7b-instruct` (current default and verified working)
- Optional Model: `llama3.1:8b-instruct-fp16` (requires additional setup; may encounter Windows-specific issues)
- Schema: YAML + Pydantic v2
- Metrics: sacrebleu, rouge-score, sentence-transformers
- CLI: Click

### Dependencies (`pyproject.toml`)

Included: langgraph, langchain, langchain-community, pydantic, pydantic-settings, pyyaml, numpy, sacrebleu, rouge-score, tenacity, click, rich, pandas, matplotlib, jupyter, ollama, torch, sentence-transformers.

Note: LangChain has deprecated `langchain_community.llms.Ollama` in favor of `langchain-ollama`. Current code uses the former but remains functional; migration is straightforward if needed.

---

## 7. Architecture Overview

- `ACERunner` initializes config, LLM client, orchestrator, evaluator, dataset utilities
- `ACEOrchestrator` builds a LangGraph `StateGraph` with nodes: generator → reflector → curator → evaluator
- `Generator` uses `LLMClient` to produce outputs from playbook context
- `Reflector` scores outputs and suggests improvements
- `Curator` updates the playbook, increments version, maintains history
- Metrics are computed and results saved under `outputs/`

LangGraph Notes (current):
- In-memory `MemorySaver` used
- `thread_id` provided at `graph.invoke` via `config={"configurable": {"thread_id": f"ace_iteration_{i}"}}`

---

## 8. Playbook Schema (Implemented)

- Models: `Heuristic`, `FewShotExample`, `PerformanceMetrics`, `HistoryEntry`, `PlaybookContext`, `PlaybookMetadata`, `Playbook`
- YAML persistence: `to_yaml`, `from_yaml`
- Base template at `agentic_context_engineering/playbook_schema/base_playbook.yaml`

---

## 9. LLM Client (Current Behavior)

- `LLMClient` wraps Ollama model usage for all agents
- GPU verification and VRAM reporting via PyTorch
- Health check invokes a minimal prompt
- Windows: avoid emojis/special glyphs in logs to prevent `UnicodeEncodeError`

Model Notes:
- Default: `mistral:7b-instruct` (verified with GPU; good speed and stability)
- Optional: `llama3.1:8b-instruct-fp16` (works with additional setup; may require more VRAM and careful Windows handling)

---

## 10. Evaluation

- Dataset generation: `DatasetGenerator` produces Python technical Q&A
- Evaluator: computes BLEU/ROUGE/EM/Semantic similarity
- Results: saved as `outputs/ace_run_*.json` and `outputs/playbook_v*.yaml`

---

## 11. Testing

- Lightweight component checks provided (`test_ace_system.py`, `simple_test.py` used during development)
- Users can generate datasets and run ACE iterations end-to-end

---

## 12. CLI and Config (Current Defaults)

### Config (`agentic_context_engineering/configs/default.yaml`)

```yaml
llm:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model: "mistral:7b-instruct"

  parameters:
    temperature: 0.7
    max_tokens: 2000
    top_p: 0.9
    num_ctx: 8192
    repeat_penalty: 1.1

  gpu:
    num_gpu: 1
    gpu_memory_fraction: 0.9
    enable_kv_cache: true

ace:
  max_iterations: 10
  convergence_threshold: 0.05
  reflection_batch_size: 3
  early_stopping_patience: 2

evaluation:
  metrics: ["bleu", "rouge", "exact_match", "semantic_similarity"]
  holdout_ratio: 0.2
  manual_review_samples: 10

performance:
  max_concurrent_requests: 1
  warm_start: true
  log_vram_usage: true

logging:
  level: "INFO"
  save_intermediate: true
  output_dir: "./outputs"
  structured_format: "json"
```

### CLI Notes (current)

- `run` options: `--iterations/-i`, `--tasks/-t`, `--playbook/-p`, `--output/-o`, optional `--config/-c` (defaults internally)
- `check-gpu` name is hyphenated: `python -m agentic_context_engineering.cli check-gpu`
- Progress UI simplified for Windows console compatibility

---

## 13. Setup & Verification (Adjusted)

- Install Ollama and pull a working model. Recommended: `ollama pull mistral:7b-instruct`
- Optional: `ollama pull llama3.1:8b-instruct-fp16`
- Verify GPU + Ollama with `python -m agentic_context_engineering.cli check-gpu`
- Windows: avoid emojis in console; the project already strips them from verification output

Example run:
```
python -m agentic_context_engineering.cli run --iterations 1 --tasks iteration_set.json
```

---

## 14. Risks and Mitigations (Observed)

- VRAM usage: OK with Mistral 7B; Llama 3.1 8B FP16 fits on 16–17 GB but may require careful environment setup
- LangChain deprecation warnings for Ollama: optional migration to `langchain-ollama`
- Windows console encoding: handled by removing emoji/special glyphs
- Git playbook commits may fail if repo is not initialized: playbook YAMLs are still saved under `outputs/`

---

## 15. Out of Scope (Same as original)

- Multi-agent collaboration beyond Gen/Ref/Cur
- RL-based curator
- Real-time user feedback
- Multi-modal inputs
- Production web UI
- Fine-tuning model weights
- Distributed multi-GPU
- Remote API endpoints

---

## 16. Integration Guidance (Project 2)

- Export playbook context via CLI `export_context` (JSON)
- Consume system prompt, heuristics, and examples in downstream agent
- Feed back new conversation data as tasks for future ACE iterations

---

## 17. Timeline (Reference)

- Setup: 0.5–1 day
- Core Development: ~5–6 days
- Evaluation & Iteration: ~2–3 days
- Documentation & Polish: ~1 day

---

Document Owner: Egemen Çankaya
Model Default: `mistral:7b-instruct` (Ollama)
Optional Model: `llama3.1:8b-instruct-fp16` (requires setup)
