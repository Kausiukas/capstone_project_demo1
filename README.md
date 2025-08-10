# 7-Agent Layers Demo — Self Development Session (LLM + RAG + Memory)

This repository hosts a focused demonstration of a 7-layer agent architecture with an emphasis on self-development sessions, retrieval-augmented generation (RAG), and long-term memory.

Repository: [Kausiukas/capstone_project_demo1](https://github.com/Kausiukas/capstone_project_demo1)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the self-development session demo
python scripts/self_dev_session.py
```

Artifacts and logs are stored under `results/` (e.g., `results/chat_sessions/`).

## Repository Layout

- `7_agent_layers/`
  - `LVL_1` … `LVL_7`: Each layer includes `layer#.md`, `runbook.md`, `tasklist.md`, `mesh.md`, and `health_checks.md` (plus layer-specific docs).
  - Cross-layer docs: `file_map.md`, `mesh_map.md`, `directory_requirements.md`, `readme.txt`.
- `scripts/`
  - `self_dev_session.py`: Orchestrates a guided self-development iteration across layers.
  - `health_probe.py`: Lightweight environment/health probe.
- `results/`
  - `chat_sessions/.gitkeep`, `llm_suggestions/.gitkeep`: Output directories used by runs.

## Layers Overview

The detailed specification for each layer lives alongside the code/docs in `7_agent_layers/LVL_#/`. Use the `layer#.md` documents for the authoritative design and behaviors.

- Layer 1 — Foundation & Tasks
  - Spec: `7_agent_layers/LVL_1/layer1.md`
  - Companion docs: `tasklist.md`, `runbook.md`, `mesh.md`, `health_checks.md`.
- Layer 2 — Tools & Interfaces
  - Spec: `7_agent_layers/LVL_2/layer2.md`
  - Companion docs as above.
- Layer 3 — Planning & Goals
  - Spec: `7_agent_layers/LVL_3/layer3.md`
  - Goals: `7_agent_layers/LVL_3/goals.md`, `7_agent_layers/LVL_3/goals.yaml`.
- Layer 4 — Orchestration & Mesh
  - Spec: `7_agent_layers/LVL_4/layer4.md`.
- Layer 5 — Self-Development & Learning
  - Spec: `7_agent_layers/LVL_5/layer5.md`, `7_agent_layers/LVL_5/self_dev_session.md` (if present).
- Layer 6 — Memory & Knowledge Management
  - Spec: `7_agent_layers/LVL_6/layer6.md`.
- Layer 7 — Governance, Reporting & Health
  - Spec: `7_agent_layers/LVL_7/layer7.md`.

Complementary cross-layer maps:
- `7_agent_layers/file_map.md` — file-level catalog
- `7_agent_layers/mesh_map.md` — interfaces and data flows
- `7_agent_layers/directory_requirements.md` — required structure

## Self-Development Session

`scripts/self_dev_session.py` drives a structured iteration where the agent reflects on goals, evaluates current capabilities, proposes improvements, and records outcomes. Expected outputs include session logs and updated artifacts in `results/`.

Run:
```bash
python scripts/self_dev_session.py
```

You can run multiple sessions and compare outputs to track capability growth over time.

## LLM + RAG

The demo is designed to integrate with an LLM backend and a vector store for retrieval-augmented generation:

- Embeddings + vector store: pluggable; typical setups use PostgreSQL + pgvector
- Retriever: semantic search over long-term memory
- Generator: LLM completes tasks using retrieved context

Configure provider credentials and endpoints via environment variables as appropriate for your LLM stack (e.g., `OPENAI_API_KEY` or compatible provider keys).

## Long-term Memory

Long-term memory is backed by a vector store. Common configuration:

- Database connection string via `DATABASE_URL`, e.g.
  - `postgresql://USER:PASS@HOST:PORT/DBNAME`
- Vector extension: `pgvector` (if using PostgreSQL)

While this repository focuses on the layered demo and session flow, it is compatible with a pgvector-backed memory if you supply a reachable database and embeddings provider.

## Connect to a Dockerized API

If you run a separate, containerized API that provides LLM/RAG services:

1. Start your API as usual (e.g., `docker compose up -d`).
2. Ensure it’s reachable from your machine (e.g., `http://localhost:8000`).
3. Provide its base URL to your scripts or environment, e.g.:
   - Set `API_URL` (or equivalent expected by your stack):
     ```bash
     set API_URL=http://localhost:8000
     ```
   - Or pass as an argument if your wrapper supports it.

If your API exposes vector operations, also set `DATABASE_URL` to point at your vector DB instance.

## License

MIT License

