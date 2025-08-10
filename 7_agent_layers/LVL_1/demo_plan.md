# L1 Demo GUI Plan (Streamlined for Stakeholders)

version: 1.0
last_updated: 2025-08-10
scope: MVP UI to operate the agent, review progress, and approve LLM suggestions

## Goals (demo-ready, minimal)
- Present a working Streamlit UI that:
  - Shows a main Chat to query system status and run sessions
  - Displays latest self-development reports and suggestions
  - Allows human-in-the-loop approval/denial/edit of suggestions
  - Provides per-layer pages and a global mesh view (read-only)
  - Shows health for DB/Ollama/API
- Backend deployed via Docker Compose; UI points to API_BASE.

## Success Criteria (pass/fail)
- Chat page can trigger self_dev_session and show result path
- Reports page renders latest 7_agent_layers/development_session_report_*.md
- Suggestions page loads latest results/llm_suggestions/*.jsonl and writes approved items to LVL_{N}/tasklist.md
- Health page green for pgvector + Ollama; metrics populated
- Mesh page renders 7_agent_layers/mesh_map.md (global) and each LVL_{N}/mesh.md (local)
 - Hosted demo on Streamlit Cloud using API_BASE + DEMO_API_KEY secrets
 - GitHub repo structured with clear README, deploy docs, and CI for Docker/image lint

## Dependencies
- Backend: src/mcp_server_enhanced_tools.py running (Docker Compose with Postgres+pgvector and Ollama)
- Vars: API_BASE, DATABASE_URL, OLLAMA_BASE_URL, OLLAMA_MODEL, optional OPENAI_API_KEY

## How to Run the UI (local)
- Set API base URL for the UI session:
  - PowerShell: `$Env:API_BASE = "http://127.0.0.1:8000"`
- Start Streamlit with the Python app (not the .md file):
  - `streamlit run web\streamlit_app.py`

Note: Do not run this `.md` file with Streamlit. Streamlit only accepts `.py` files.

## Status Checklist
- [x] Streamlit UI available with pages: Chat, Reports, Suggestions, Layers, Mesh, Health, Tools/APIs (`web/streamlit_app.py`)
- [x] UI reads `API_BASE` from environment and shows it in sidebar
- [x] Backend running via Docker Compose with demo endpoints:
  - [x] `POST /agent/answer`
  - [x] `GET /performance/metrics`
  - [x] `GET /tools/registry/health`
- [x] Reports page renders latest `development_session_report_*.md`
- [x] Suggestions page loads latest `results/llm_suggestions/*.jsonl` and appends approvals to `LVL_{N}/tasklist.md`
- [x] Mesh page renders `7_agent_layers/mesh_map.md`
- [x] Chat works and can run a 2‑min local `self_dev_session.py` from the UI (stdout shown; latest report filename displayed)
- [x] Health page includes detailed pgvector and Ollama checks (enriched `/health` implemented)
- [x] Chat uses basic RAG + memory via pgvector; history persisted
- [ ] LLM Suggestions page populated by an automated suggestions job
- [ ] Streamlit Cloud deployment live (URL shared)
- [ ] GitHub repository structure finalized and CI green

## Pages (must-have only)
1) Chat
- Prompt box; send to simple API: POST /agent/answer (fallback: display canned response)
- Buttons: Start/Stop session (spawns scripts/self_dev_session.py via backend helper endpoint or local process note)
- Recent status: last report filename

2) Reports
- List and view 7_agent_layers/development_session_report_*.md
- Quick metrics: cycles, actions, LLM tokens, readiness snapshot

3) Suggestions (Human-in-the-loop)
- Load latest results/llm_suggestions/session_*.jsonl
- Table with layer, path, rationale, steps
- Approve/Deny/Edit; on Approve: append to 7_agent_layers/LVL_{layer}/tasklist.md as "- [ ] ... (source: LLM {ts})"
 - Note: if empty, run the suggestions generation job (see Next Priorities)

4) Layers
- Tabs for L1-L7
- Render LVL_{N}/layer{n}.md top and LVL_{N}/tasklist.md (done/remaining counts)

5) Mesh
- Render 7_agent_layers/mesh_map.md (table)
- Links to LVL_{N}/mesh.md

6) Health
- Call GET /health; GET /performance/metrics; optional button to run scripts/health_probe.py
- Show DB/Ollama/GPU status

7) Tools/APIs (read-only)
- Show endpoints and minimal examples; list registry health if available (/tools/registry/health)

## Build Plan (4-6 hours focused)
- Hour 0-1: Streamlit shell
  - Project web/streamlit_app.py (or streamlit_app/ with multipage)
  - Global navbar; read API_BASE from env
- Hour 1-2: Chat + Health
  - Chat posting to /agent/answer (fallback text if 404)
  - Health cards: /health, /performance/metrics
- Hour 2-3: Reports + Suggestions
  - Reports: file browse + markdown render
  - Suggestions: load JSONL; approve -> append to tasklist
- Hour 3-4: Layers + Mesh
  - Tabs for L1-L7; render layer{n}.md and tasklist.md
  - Global mesh table render
- Hour 4-5: Tools/APIs page + polish
  - Show key endpoints and sample curl/PowerShell
  - Minimal theming (title/logo)
- Hour 5-6: Demo script dry run and fixes

## Repository Structure (target)
```
root/
  README.md (quick start + demo link)
  web/ (Streamlit app)
  src/ (FastAPI, tools, memory)
  scripts/ (sessions, indexers)
  deployment/ (docker, compose, cloud docs)
  7_agent_layers/ (docs, tasklists, mesh)
  results/ (reports, llm_suggestions, chat_sessions)
  .github/workflows/ (ci.yml: lint, build image)
  .streamlit/ (config.toml if needed)
```

## Streamlit Cloud Deployment
- Secrets: `API_BASE`, `DEMO_API_KEY` (and optional `THEME`)
- Command: `streamlit run web/streamlit_app.py`
- Post‑deploy check: Chat works, Health green, Suggestions loads

## Demo Script (5-7 min)
1) Health is green; small local model (llama3.2:3b) running
2) Chat asks status -> system reply; start 2-min session
3) Open last report; show cycles, actions, LLM tokens
4) Open Suggestions; approve one small test/doc item -> it appears in target layer tasklist
5) Open Layers L5.1 page (self-dev docs); show workflow
6) Open Mesh; show one-line-per-layer map

## Risk Controls
- If /agent/answer unavailable, fallback to read-only Chat with guidance
- If suggestions empty, show tip to run session with --use-llm --preload-llm
- If health partial (no GPU), display warning but continue (CPU ok)

## Commands (quick start)
- Backend (Docker Compose): bring up API + DB + Ollama (if 11434 busy, use native)
- UI (local dev): streamlit run web/streamlit_app.py
- Session run: venv\\Scripts\\python scripts\\self_dev_session.py --duration-min 2 --sleep 0.2 --use-llm --preload-llm

### API helpers
- Health (enriched): `Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 6`
- Run session via API: `Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8000/admin/run_session -ContentType 'application/json' -Body '{"duration_min": 2, "sleep": 0.2, "use_llm": true, "preload_llm": true, "max_tests_per_loop": 1}'`

## Out of Scope (defer)
- Voice I/O, image preview polish
- Full orchestrator UI; advanced dashboards
- Auth; role-based approvals

## Next Priorities (1–3 days)
- L5 Orchestration: implement suggestions generation job producing `results/llm_suggestions/session_*.jsonl` (endpoint + schedule); link it from UI
- L4 Reasoning: improve chat grounding and citations (see L4 tasklist); add follow‑up suggestions in UI
- L6 Memory: switch to Ollama embeddings and add per‑file chunking/tags; nightly incremental indexer
- L7 Infra: finalize GitHub repo (README, LICENSE, issues templates), add CI (`.github/workflows/ci.yml`) to lint and build Docker image; write Streamlit Cloud deploy doc
- L1 UI: polish Suggestions page (filter by layer, search), add “Run suggestions job” button; small UX improvements (copy link to last report)
