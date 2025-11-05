# Agentic Context Engineering (ACE)

[![Release](https://img.shields.io/github/v/release/EgeCankaya/Agentic-Context-Engineering)](https://img.shields.io/github/v/release/EgeCankaya/Agentic-Context-Engineering)
[![Build status](https://img.shields.io/github/actions/workflow/status/EgeCankaya/Agentic-Context-Engineering/main.yml?branch=main)](https://github.com/EgeCankaya/Agentic-Context-Engineering/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/EgeCankaya/Agentic-Context-Engineering/branch/main/graph/badge.svg)](https://codecov.io/gh/EgeCankaya/Agentic-Context-Engineering)
[![Commit activity](https://img.shields.io/github/commit-activity/m/EgeCankaya/Agentic-Context-Engineering)](https://github.com/EgeCankaya/Agentic-Context-Engineering)
[![License](https://img.shields.io/github/license/EgeCankaya/Agentic-Context-Engineering)](https://github.com/EgeCankaya/Agentic-Context-Engineering)

**Agentic Context Engineering (ACE)** — Evolving Contexts for Self-Improving Language Models

A framework that enables self-improvement of large language models through context evolution rather than model fine-tuning. The system iteratively improves performance by running a three-phase loop: **Generator → Reflector → Curator**.

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA RTX 4070 Ti Super (16GB VRAM) or equivalent
- **RAM**: 32GB DDR5
- **Python**: 3.9+
- **CUDA**: Version 11.8 or higher

### Installation

1. **Install Ollama and pull Llama 3.1 8B:**
```bash
# Install Ollama (Windows)
winget install Ollama.Ollama

# Pull Llama 3.1 8B Instruct (~16GB download)
ollama pull llama3.1:8b-instruct-fp16

# Verify installation
ollama list
```

2. **Install ACE:**
```bash
git clone https://github.com/EgeCankaya/Agentic-Context-Engineering.git
cd Agentic-Context-Engineering
pip install -e .
```

3. **Verify setup:**
```bash
python verify_setup.py
```

### Basic Usage

```python
from agentic_context_engineering import ACERunner

# Initialize ACE system
runner = ACERunner()

# Run 5 iterations
results = runner.run_iterations(num_iterations=5)

# Check results
print(f"Completed {results['total_iterations']} iterations")
print(f"Final playbook version: {results['final_playbook'].version}")
```

### CLI Usage

```bash
# Run ACE iterations
python -m agentic_context_engineering.cli run --iterations 10

# Evaluate a playbook
python -m agentic_context_engineering.cli evaluate --playbook outputs/playbook_v1.5.0.yaml

# Compare two playbook versions
python -m agentic_context_engineering.cli diff --from-version 1.0.0 --to-version 1.5.0

# Check system health
python -m agentic_context_engineering.cli check-gpu
```

## 📖 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Evaluation Dataset                     │
│              (Dev / Iteration / Held-Out)                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              ACE Iteration Controller                    │
│              (ace_runner.py)                             │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 LangGraph Orchestrator                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │Generator │ -> │Reflector │ -> │ Curator  │         │
│   │  Agent   │    │  Agent   │    │  Agent   │         │
│   └──────────┘    └──────────┘    └──────────┘         │
│         │               │                │              │
│         └───────────────┴────────────────┘              │
│                         │                               │
│          All use Llama 3.1 8B (via Ollama)             │
└─────────────────────────┼───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Playbook Repository                         │
│   (Versioned Context + Git History)                     │
│   - system_instructions                                 │
│   - heuristics                                          │
│   - examples                                            │
│   - constraints                                         │
│   - metadata                                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          Evaluation & Visualization                      │
│   - Metrics Dashboard                                   │
│   - Before/After Comparison                             │
│   - Playbook Diff Viewer                                │
│   - VRAM Usage Tracking                                 │
└─────────────────────────────────────────────────────────┘
```

### The ACE Loop

1. **Generator**: Produces model outputs using current playbook context
2. **Reflector**: Evaluates outputs and identifies improvement areas
3. **Curator**: Updates playbook with new heuristics and examples

### Key Features

| Feature | Description |
|---------|-------------|
| **Playbook Schema** | Structured, versioned representation of model context with Pydantic validation |
| **Local Inference** | Zero-cost operation using Llama 3.1 8B via Ollama |
| **Iteration Loop** | Automated generation→reflection→curation cycles with convergence detection |
| **Evaluation System** | Multi-metric evaluation (BLEU, ROUGE, task-specific accuracy) |
| **Version Control** | Git-based playbook versioning with semantic versioning |
| **CLI Interface** | Comprehensive command-line interface for all operations |

## 🎯 Performance Benchmarks

### Hardware Requirements
- **GPU**: RTX 4070 Ti Super (16GB VRAM)
- **VRAM Usage**: ~9GB
- **Inference Speed**: 40-50 tokens/sec
- **Full Iteration**: 2-3 minutes

### Expected Improvements
- **Task Accuracy**: ≥15% improvement over baseline
- **Response Coherence**: ≥10% BLEU/ROUGE increase
- **Convergence**: Plateau within 10 iterations
- **System Reliability**: 100% iteration completion

## 📁 Project Structure

```
agentic_context_engineering/
├── agents/              # Generator, Reflector, Curator agents
├── playbook_schema/     # Pydantic models + base template
├── runners/             # ACE iteration orchestrator
├── utils/               # LLM client, metrics, versioning
├── eval/                # Evaluation dataset + evaluator
├── configs/             # YAML configurations
├── cli.py               # Command-line interface
└── tests/               # Unit and integration tests
```

## 🔧 Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
llm:
  provider: "ollama"
  model: "llama3.1:8b-instruct-fp16"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 2000

ace:
  max_iterations: 10
  convergence_threshold: 0.05
  reflection_batch_size: 3
  early_stopping_patience: 2

evaluation:
  metrics: ["bleu", "rouge", "exact_match", "semantic_similarity"]
  holdout_ratio: 0.2
```

## 📊 Example Results

### Before ACE (Baseline)
**Input:** "How do I handle retries in Python API calls?"

**Output:**
```
You can use a try-except block to catch exceptions and retry the API call.
```

### After ACE (Iteration 8)
**Input:** "How do I handle retries in Python API calls?"

**Output:**
```
Use the `tenacity` library for robust retry logic with exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_api():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()
```

This will retry up to 3 times with exponential backoff (2s, 4s, 8s).

**Documentation:** https://tenacity.readthedocs.io/
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=agentic_context_engineering --cov-report=html

# Run specific test categories
pytest tests/test_playbook.py -v
pytest tests/test_llm_client.py -v
```

## 📚 Documentation

- **API Reference**: [Documentation](https://EgeCankaya.github.io/Agentic-Context-Engineering/)
- **Paper**: [Agentic Context Engineering: Evolving Contexts for Self‑Improving Language Models](https://arxiv.org/abs/2510.04618)
- **Ollama Setup**: [Ollama Documentation](https://ollama.com/docs)
- **Integration Guide**: `docs/integration_guide.md`
- **Project 2 Checklist**: `docs/project2_integration_checklist.md`

## 🧩 Using ACE in Your Project

1. **Install the package**
   ```bash
   pip install "agentic-context-engineering @ git+https://github.com/EgeCankaya/Agentic-Context-Engineering.git"
   ```
2. **Load the latest playbook**
   ```python
   from agentic_context_engineering import Playbook

   playbook = Playbook.from_yaml("outputs/playbook_v1.2.0.yaml")
   bundle = playbook.export_context_for_rag({"course": 12})
   ```
3. **Generate answers with retrieved context**
   ```python
   from agentic_context_engineering import LLMClient
   from agentic_context_engineering.agents import Generator

   llm_client = LLMClient()
   generator = Generator(llm_client)
   response = generator.generate_with_context(question, playbook, retrieved_docs)
   citations = generator.extract_citations(response, retrieved_docs)
   ```
4. **Log conversations for ACE feedback**
   ```python
   from agentic_context_engineering import ConversationLogger

   logger = ConversationLogger(output_dir="outputs/conversations")
   session_id = logger.start_session(user_id="student")
   logger.log_turn(session_id, question, response, retrieved_docs)
   dataset = logger.export_for_ace()
   ```
5. **Run improvement cycles**
   ```python
   from agentic_context_engineering import ACERunner

   runner = ACERunner()
   runner.run_iterations(num_iterations=3, tasks=[item["input"] for item in dataset])
   ```

📁 Check `examples/external_rag_integration/` for a runnable demo that ties these steps together (simple RAG bot, conversation logger, and ACE trigger script).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Foundation**: Based on the paper "Agentic Context Engineering: Evolving Contexts for Self‑Improving Language Models"
- **Model**: Llama 3.1 8B Instruct by Meta AI
- **Framework**: LangGraph for agent orchestration
- **Local Inference**: Ollama for zero-cost operation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/EgeCankaya/Agentic-Context-Engineering/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EgeCankaya/Agentic-Context-Engineering/discussions)
- **Email**: egemencankaya14@gmail.com

---

**Ready to evolve your LLM contexts?** 🚀

```bash
python verify_setup.py  # Check your system
python -m agentic_context_engineering.cli run --iterations 5  # Start iterating!
```
