# Runbook – Layer 5: Tools & API

## Common symptoms
- Chain failures; high step P95; frequent retries; schema mismatches

## Diagnostics
- GET /tools/registry/health (if enabled)
- Inspect exec_traces in Memory; check chain_success and fallback_rate

## Remedies
- Update adapters and IO validators; adjust retries/backoff; rotate fallbacks
- Tune selection policy (latency/success/cost)

---

## 5.1 Self-Development Orchestrator
- Purpose: automate short, safe, iterative development cycles guided by task dashboards and LLM suggestions with human approval.
- Inputs: dashboard JSON (`scripts/tasklog_dashboard.py --auto --json`), LLM provider (OpenAI/Ollama), reviewer approvals.
- Outputs: suggestions JSONL, created scaffolds/tests, session reports (`7_agent_layers/development_session_report_*.md`).

### Operations
1) Pre-flight probe (`scripts/health_probe.py`), optionally `--preload-llm`.
2) Run traces, refresh dashboard, compute recommendations.
3) Generate LLM suggestions; save under `results/llm_suggestions/`.
4) Human reviews suggestions; approved items appended to `LVL_{N}/tasklist.md`.
5) Apply safe scaffolds/tests; run targeted pytest; count checkbox flips as actions.
6) Persist report with bottlenecks, tasks before/after, LLM usage, readiness snapshot.

### Recovery
- If suggestions timeout: increase `OLLAMA_CHAT_TIMEOUT` (e.g., 120), use smaller model, or force OpenAI with `LLM_PROVIDER=openai`.
- Check `results/llm_suggestions/error_last.txt` for last error.
