# Project 2 Integration Checklist

Use this checklist to wire the IBM Gen AI Companion (Project 2) into the ACE framework.

## Setup

- [ ] Clone or update the ACE repository locally.
- [ ] Install ACE as a dependency in the companion project (`pip install -e ../Agentic-Context-Engineering`).
- [ ] Verify Ollama or chosen LLM backend is operational.
- [ ] Copy `.env.example` (if needed) and configure API keys or base URLs.

## Playbook Integration

- [ ] Load the latest ACE playbook (`Playbook.from_yaml(...)`).
- [ ] Call `playbook.export_context_for_rag()` during start-up to hydrate prompts.
- [ ] Refresh playbook after every ACE cycle (reload YAML file).

## Retrieval & Generation

- [ ] Ingest IBM course materials into a vector store (Chroma/Qdrant).
- [ ] Implement retrieval function returning documents shaped like `{id, content, metadata}`.
- [ ] Use `Generator.generate_with_context()` to combine ACE prompt + retrieval results.
- [ ] Capture citations with `Generator.extract_citations()`.

## Conversation Logging

- [ ] Instantiate `ConversationLogger` at application startup.
- [ ] For each user turn, log question, answer, retrieved docs, and annotations.
- [ ] Store logs in `outputs/conversations/` (ensure folder is git-ignored).
- [ ] Periodically export logs via `logger.export_for_ace()`.

## ACE Improvement Loop

- [ ] Schedule ACE runs (e.g., nightly or after N conversations).
- [ ] Provide exported conversation dataset as `tasks` to `ACERunner.run_iterations()`.
- [ ] Monitor `results["final_playbook"]` and copy YAML to companion project.
- [ ] Commit/new tag for each significant playbook milestone.

## Evaluation & QA

- [ ] Build a 100-question evaluation set covering all 16 courses.
- [ ] Use `EducationalEvaluator` for pedagogy and grounding scores.
- [ ] Compare baseline vs. ACE-enhanced metrics (accuracy, citation rate, deflection).
- [ ] Review flagged conversations (scores < 0.5) and adjust heuristics.

## Deployment Hygiene

- [ ] Update project README with ACE integration notes and usage instructions.
- [ ] Document fallback behaviour when LLM backend is offline.
- [ ] Ensure personal/course data complies with IBM terms of service.
- [ ] Plan for regression testing before releasing updated playbooks.
