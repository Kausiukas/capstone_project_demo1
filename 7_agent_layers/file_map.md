# File Map (minimal 7_agent_layers demo)

Goal: keep the repository focused on the 7 Agent Layers demo with a lightweight Streamlit UI that connects to an external API. Remove backend and docker files from this repo.

## Keep
- Root
  - `README.md` – purpose, demo instructions (UI + external API), Streamlit Cloud notes
  - `requirements.txt` – only UI deps: `streamlit`, `requests`
  - `.gitignore`
- UI
  - `web/streamlit_app.py` – pages: Chat, Reports, Suggestions, Layers, Mesh, Health, Tools (calls external API via `API_BASE`)
- Demo content
  - `7_agent_layers/**` – docs, mesh, tasklists, `LVL_*/`
  - `7_agent_layers/file_map.md` (this file)
  - `7_agent_layers/repository_population_plan.md` (curation plan)
- Outputs (empty placeholders committed as `.gitkeep`)
  - `results/llm_suggestions/.gitkeep`
  - `results/chat_sessions/.gitkeep`
  - `7_agent_layers/development_session_report_examples/` (optional sample report)

## Remove (from this repo)
- `src/**`, `deployment/**`, `scripts/**`, `.github/workflows/**` (backend/docker/CI live in the API repo)
- any `pgvector_data/**`, caches (`__pycache__`, `.pytest_cache`, `node_modules`, etc.)

## Runtime expectations
- UI requires these environment variables on Streamlit Cloud/local:
  - `API_BASE` – URL of external API
  - `DEMO_API_KEY` – passed as `X-API-Key`

## Run locally
```
pip install -r requirements.txt
set API_BASE=http://127.0.0.1:8000
set DEMO_API_KEY=demo_key_123
streamlit run web/streamlit_app.py
```

