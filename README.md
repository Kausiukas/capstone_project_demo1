# LangFlow Connect MVP - Integrated Dashboard

## 🎯 Project Overview

This is a **Capstone Project** demonstration of the LangFlow Connect MCP (Model Context Protocol) server with integrated Content Preview and Performance Monitoring systems.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run MCP server
python src/mcp_server_enhanced_tools.py

# Run integrated dashboard
streamlit run streamlit_app_integrated.py
```
## PostgreSQL + pgvector (Docker)

Prereqs:
- Docker Desktop on Windows with WSL2 backend enabled

Start/stop:
- Start DB: `docker compose up -d`
- Stop DB: `docker compose down`
- Logs: `docker compose logs -f db`
- Verify: `python scripts/verify_pgvector.py`

Automation:

```bash
# Ensure DB is up, wait for readiness, ensure pgvector, and print DATABASE_URL
python scripts/ensure_pgvector.py --timeout 120

# Optionally set it in the current process and see the value
python scripts/ensure_pgvector.py --set-env

# Then run tests that need the DB
pytest -q tests/phase1/test_pgvector_basic.py
```

Details:
- Data persists in the `pgdata` Docker volume
- Config via `.env` (defaults provided in docs); connection string `DATABASE_URL`
- Default port mapping `5432:5432`. If port is in use, change left side to `5433:5432` and set `DATABASE_URL=postgresql://app:app@localhost:5433/appdb`

Troubleshooting:
- Port 5432 in use → change mapping and `DATABASE_URL`
- WSL high memory → `wsl --shutdown` in PowerShell to reclaim RAM
- Resource limits → Docker Desktop → Settings → Resources


## 🛠 Features

### Core Tools
- **5 Core Tools**: ping, read_file, list_files, get_system_status, analyze_code
- **Universal File Access**: Local, GitHub, and HTTP file support
- **Web Interface**: Streamlit-based unified dashboard
- **API Access**: RESTful API for programmatic access

### Content Preview System
- **Syntax Highlighting**: Support for 20+ programming languages
- **Markdown Rendering**: Full markdown to HTML conversion
- **Image Preview**: Base64 encoding for inline display
- **Batch Processing**: Preview multiple files simultaneously
- **File Analysis**: Automatic type detection and capabilities

### Performance Monitoring
- **Real-time Metrics**: Response times, success rates, error counts
- **System Monitoring**: CPU, memory, disk usage tracking
- **Performance Alerts**: Automated alerting for issues
- **Health Monitoring**: Comprehensive system health checks
- **Tool-specific Metrics**: Individual tool performance tracking

## 📊 Dashboard Sections

1. **🏠 Dashboard** - Overview and quick actions
2. **🛠️ Tool Testing** - Interactive tool execution
3. **👁️ Content Preview** - File preview and analysis
4. **📊 Performance Monitoring** - Real-time metrics and alerts
5. **🗺️ Topology** - Live ports/processes/env and DB/pgvector status
5. **📚 API Docs** - Complete API documentation
6. **🔧 System Status** - System health and configuration

## 🔧 Configuration

The dashboard automatically connects to the deployed API at:
`https://capstone-project-api-jg3n.onrender.com`

You can change the API URL in the sidebar configuration.

## 🧰 Utilities

- `scripts/ports_status.py`: list listening ports and owning processes; helpful for resolving conflicts (8000/8501/5433)
- `scripts/find_free_port.py`: find an available local TCP port in a range
- `scripts/topology_probe.py`: query the API `/admin/topology` endpoint and pretty-print results
- `scripts/warmup_api.py`: warm up key endpoints (`/health`, `/performance/*`, `/db/*`) to prime caches/DB

Examples:

```bash
python scripts/ports_status.py --common
python scripts/find_free_port.py --start 8000 --end 8100
python scripts/topology_probe.py --api http://127.0.0.1:8000 --key demo_key_123
python scripts/warmup_api.py --api http://127.0.0.1:8000 --key demo_key_123
```

## 📄 License

MIT License

## 🎯 Capstone Project Status

✅ **Complete** - All systems integrated and functional
- Core MCP tools operational
- Content Preview System active
- Performance Monitoring active
- Unified dashboard deployed
- Universal file access working
- Real-time metrics collection
- Comprehensive error handling

---

## 🧱 Seven-Layer Architecture Overview

### Layers at a glance
- **Layer 1 – Human Interface (95%)**: Streamlit dashboard with content preview, performance monitoring, tool testing, and full API docs. Chat (stream + history), semantic search, document ingestion.
- **Layer 2 – Information Gathering (90%)**: Collectors (files, HTTP, system), normalizers, synthesizer, and validators. Enhanced ingestion powered by local Ollama; pgvector indexing.
- **Layer 3 – Structure, Goals & Behaviors (75%)**: Goal management, website optimization instrumentation, local resource learning, ethical scaffolding, personality stubs.
- **Layer 4 – Agent Brain (45%)**: LLM adapter (Ollama + mock) implemented; reasoning, planning, decision stubs wired with sketches for next phases.
- **Layer 5 – Tools & API (75%)**: Tool registry, orchestrator, execution engine, MCP tools (ping/read/list/status/analyze), security and performance hooks.
- **Layer 6 – Memory & Feedback (85%)**: pgvector-backed memory, short‑term window with autosummary, feedback-weighted ranking, chat history, endpoints.
- **Layer 7 – Infra, Scaling & Security (80%)**: Docker/Compose (API, Postgres+pgvector, Ollama), health checks, security headers, API keys, monitoring; prod deploy paths.

### System diagram
```mermaid
graph TB
  U[User] --> L1[Layer 1 UI]
  L1 --> API[FastAPI Backend]

  subgraph L2-L6 Core
    L2[Layer 2 Info Gathering] --> DB[(PostgreSQL + pgvector)]
    L6[Layer 6 Memory] --> DB
    L5[Layer 5 Tools/API] --> MCP[MCP Tools]
    L4[Layer 4 Brain] --> L5
    L4 --> L6
    L2 --> L6
  end

  API --> L2
  API --> L4
  API --> L5
  API --> L6
  API --> MON[Monitoring]
  API --> OLL[Ollama LLM]

  L7[Layer 7 Infra/Security] -. manages .- API
  L7 -. manages .- DB
  L7 -. manages .- OLL
  L7 -. monitors .- MON
```

### Representative use cases
- **Chat with memory (local models)**: Stream chat via UI or API; short-term window + long-term pgvector memory; feedback tunes retrieval.
- **Document understanding & semantic search**: Ingest PDFs/markdown/code with enhanced extraction (Ollama). Query semantically via `/memory/query`.
- **Code and repo review via tools**: Use MCP tools (`read_file`, `analyze_code`) from the dashboard to inspect repositories and produce briefs.
- **Ops visibility & health**: Monitor response times, success rates, and DB/vector health; run warmups and topology probes.
- **Automated learning signals**: Layer 3 generates learning goals from local resources; Layer 6 stores outcomes for future improvement.

### Per-layer highlights (most important)
- **L1**: Unified dashboard + dynamic docs; real-time metrics; streaming chat with session history.
- **L2**: Robust collectors + normalizers; AI-enhanced ingestion; structured storage with provenance.
- **L3**: Goal/OKR scaffolding with website optimization and learning loops from local resources.
- **L4**: Pluggable LLM adapter (Ollama) ready; reasoning/planning/decision flows outlined for next phase.
- **L5**: Tool registry + orchestrator with retries/fallbacks; secure tool execution; health and metrics.
- **L6**: Memory CRUD + semantic search; feedback-adjusted ranking; short-term autosummary.
- **L7**: Compose stack (API, Postgres, Ollama), security headers, API keys, health checks, deployments.

### Near-term roadmap
- **Reasoning & Planning (L4)**: Evidence-tracked chain-of-thought; plan validation; cost/latency-aware decisions.
- **Advanced Orchestration (L5)**: DAG/parallel flows; result aggregation; budget policies.
- **Consolidation & Learning (L6)**: Nightly consolidate/summarize; HNSW; learning analytics.
- **Security & Scaling (L7)**: Token-bucket rate limits, key rotation, auto-scaling, SLOs/error budgets.
- **UI/UX (L1)**: Voice I/O, multilingual UI, accessibility improvements.

### How this system fits together
- UI (L1) invokes API, which routes to Info Gathering (L2), Tools (L5), Memory (L6), and Brain (L4).
- Memory (L6) powers RAG and chat context via pgvector; feedback closes the loop.
- Tools (L5) give the Brain (L4) reliable, observable capabilities.
- Infra (L7) secures and scales the whole stack; monitoring surfaces health/SLIs.
