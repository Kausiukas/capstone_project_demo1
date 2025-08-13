## Overview

This repository implements a 7-layer agent system with an operational FastAPI backend, Streamlit UI, Smart Scout automated tests, PostgreSQL (pgvector) memory, and Ollama LLM integration. The system supports instance identity, smart ping/tracing, tool-enabled chat, document ingestion, and RAG.

Repository: [Kausiukas/capstone_project_demo1](https://github.com/Kausiukas/capstone_project_demo1)

## Architecture (Mermaid)

```mermaid
graph LR
  U["User"] --- S["Streamlit UI (web/streamlit_app.py)"]
  S ---|HTTP| A["FastAPI Backend (api/main.py)"]
  A ---|Embeddings| O["Ollama (llama3.2:3b, mxbai-embed-large)"]
  A ---|pgvector| P["PostgreSQL (langflow-postgres)"]
  A --- I["Identity + Health + Tools"]
  S -.-> SM["Smart Ping Page"]
  U -->|Commands/Chat| S
```

## Critical Data Flow (Mermaid)

```mermaid
sequenceDiagram
  participant UI as Streamlit UI
  participant API as FastAPI
  participant OLL as Ollama
  participant DB as PostgreSQL+pgvector
  UI->>API: /ingest/files or /memory/store
  API->>OLL: /api/embeddings (mxbai-embed-large)
  OLL-->>API: embedding vector (1024-d)
  API->>DB: INSERT content + embedding (vector_store)
  UI->>API: /memory/query?query=...
  API->>DB: SELECT top-k by similarity/text
  DB-->>API: snippets
  API-->>UI: answer + used_docs
```

## Quick Start (local)

```bash
pip install -r requirements.txt
streamlit run web/streamlit_app.py --server.port 8501
```

Backend (Docker):

```bash
docker compose up -d --build api
# Ensure api container can reach ollama and postgres (bridge network)
```

Key environment variables:
- API_KEY: demo_key_123
- LLM_PROVIDER: ollama
- OLLAMA_BASE: http://<ollama_host_or_ip>:11434
- OLLAMA_CHAT_MODEL: llama3.2:3b
- OLLAMA_EMBED_MODEL: mxbai-embed-large
- DATABASE_URL: postgresql://langflow_user:langflow_password@<db_host_or_ip>:5432/langflow_connect

## Endpoints (highlights)

- Identity/Health:
  - GET `/identity`
  - GET `/health`
- Memory:
  - POST `/memory/store` (StoreBody: user_input, response)
  - GET `/memory/query?query=...&k=5`
  - GET `/memory/documents?limit=20`
- Ingestion:
  - POST `/ingest/files` (multipart form)
- Chat:
  - POST `/chat/message` (session_id, message, top_k)
  - POST `/chat/stream`
- Tools:
  - GET `/tools/list`
  - POST `/api/v1/tools/call`

## Tool-Enabled Chat

Chat can execute tools automatically or on explicit commands:

- List documents (explicit):
  - `!list_docs 20`
- Natural language trigger:
  - “what files/documents are present in the database?”

Response includes a concise list plus the full data payload. Works in `/chat/message` and `/chat/stream`.

## Smart Scout (Mesh-aware)

Run full readiness test and generate report:

```bash
python scripts/run_mesh_scout.py
```

Output: `results/scout_reports/scout_report_<timestamp>.json`

## Layer 6 (Memory) Mesh (Mermaid)

```mermaid
graph TD
  A["/memory/store"] --> E["Embed via Ollama"]
  E --> VS["vector_store (pgvector)"]
  Q["/memory/query"] --> VS
  VS --> R["snippets"]
  R --> G["LLM Generate"]
```

## LangGraph Sketch (illustrative)

```python
from langgraph.graph import Graph

g = Graph()
g.add_node("ingest")
g.add_node("embed")
g.add_node("store")
g.add_node("retrieve")
g.add_node("generate")

g.add_edge("ingest", "embed")
g.add_edge("embed", "store")
g.add_edge("retrieve", "generate")

# In practice map nodes to your concrete functions/endpoints
```

## Troubleshooting

- Ollama 404/connection: ensure models are pulled (`mxbai-embed-large`, `llama3.2:3b`) and API can reach the ollama host/IP.
- DB auth/role errors: confirm container env (`POSTGRES_USER=langflow_user`, etc.) and `DATABASE_URL` points to the actual service (avoid host.docker.internal unless intended).
- DNS issues between containers: connect `capstone_api` to the `bridge` network or use service IPs.

## Streamlit Cloud Notes

- Include `requirements.txt` with `streamlit`, `requests` and any client-side deps.
- Ensure the app entry is `web/streamlit_app.py` (or adjust).
- API URL and key should be set via Streamlit secrets or environment.

## Repository Layout (selected)

- `api/main.py`: FastAPI backend (identity, memory, tools, chat)
- `web/streamlit_app.py`: Streamlit UI
- `scripts/run_mesh_scout.py`: Mesh-aware Smart Scout launcher
- `scripts/test_critical_postgresql_embeddings.py`: critical store/retrieve tests
- `7_agent_layers/`: design specs, maps, and diagrams per layer

## License

MIT License
