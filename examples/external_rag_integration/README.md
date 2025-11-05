# External RAG Integration Example

This example project demonstrates how to embed the **Agentic Context Engineering (ACE)** framework inside an external Retrieval-Augmented Generation (RAG) chatbot. It provides a minimal, reproducible workflow that you can adapt for the IBM Gen AI Companion project.

## Folder Structure

```
examples/external_rag_integration/
├── README.md                 # This guide
├── requirements.txt          # Python dependencies
├── simple_rag_bot.py         # Minimal RAG + ACE chatbot
├── conversation_logger.py    # Script to log demo conversations
├── trigger_ace.py            # Run ACE iterations on logged data
└── sample_conversations.json # Example conversation dataset
```

## Prerequisites

- Python 3.9+
- ACE installed (see `docs/integration_guide.md`)
- Optional: Ollama with a local model (e.g., `mistral:7b-instruct`)

Install the example dependencies:

```bash
pip install -r examples/external_rag_integration/requirements.txt
```

## 1. Run the Simple RAG Bot

```bash
python examples/external_rag_integration/simple_rag_bot.py \
  --question "How do transformers use self-attention?"
```

What it does:

1. Loads the latest ACE playbook (falling back to the base template).
2. Retrieves toy documents that mimic IBM course notes.
3. Builds a combined prompt using ACE heuristics + retrieved snippets.
4. Generates an answer and prints identified citations.

## 2. Log Conversations

After running the bot, capture the Q&A turn for ACE feedback:

```bash
python examples/external_rag_integration/conversation_logger.py
```

This script:
- Creates a `ConversationLogger`
- Writes the turn to `outputs/conversations/<session_id>.json`
- Prints the ACE-ready dataset entries

## 3. Trigger ACE Improvement Cycle

```bash
python examples/external_rag_integration/trigger_ace.py \
  --iterations 2 \
  --tasks outputs/conversations/sample_dataset.json
```

This will:
- Load the logged conversations
- Run Generator → Reflector → Curator loops for the specified iterations
- Save an updated playbook under `outputs/playbook_v*.yaml`

## 4. Sample Data

`sample_conversations.json` contains a mock dataset that mirrors the IBM Gen AI curriculum. Use it to simulate ACE runs before collecting real conversations.

## Next Steps

- Replace `toy_documents` in `simple_rag_bot.py` with your LangChain or LangGraph retriever.
- Feed actual student questions into the logger.
- Schedule `trigger_ace.py` to run periodically (e.g., nightly) to keep the playbook fresh.
- Compare baseline vs. ACE-enhanced answers using the `EducationalEvaluator` introduced in Project 1.

For a deeper integration walkthrough, consult `docs/integration_guide.md` and `docs/project2_integration_checklist.md`.
