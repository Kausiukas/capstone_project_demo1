# Self-Development Session (5.1)

version: 1.0
last_updated: 2025-08-09
scope: Layer 5.1 – Self-Development Orchestrator

## Purpose
Enable iterative, time-boxed self-improvement cycles that:
- read the current task dashboard,
- generate small, safe LLM suggestions,
- keep a human approval gate,
- apply minimal scaffolds/tests,
- produce a concise report with metrics and readiness snapshot.

## System placement
- Lives under Layer 5 (Tools & API) as submodule 5.1.
- Consumes traces, tasklists, and LLM provider; produces suggestions JSONL and session reports.
- Human-in-the-loop integrates with L1 UI (review/approve/edit suggestions, run sessions, view reports).

## Script: `scripts/self_dev_session.py`
- Inputs: duration, sleep, layer/task caps, LLM provider, preload flags.
- Outputs: `results/llm_suggestions/session_*.jsonl`, `7_agent_layers/development_session_report_*.md`.
- Key behaviors:
  - Health probe (pgvector/Ollama/GPU) and optional model warmup.
  - Dashboard auto-rules to reflect completed work.
  - LLM suggestions (OpenAI or Ollama). Suggestions are not auto-applied.
  - Safe scaffolds for tests/docs to flip readiness quickly.
  - Focused pytest to keep loops fast.

## Typical usage
- Short loop (sanity):
  - `venv\\Scripts\\python scripts\\self_dev_session.py --duration-min 5 --sleep 0.2 --use-llm --preload-llm`
- L1-biased push (UI work):
  - `venv\\Scripts\\python scripts\\self_dev_session.py --duration-min 60 --sleep 0.2 --use-llm --preload-llm --max-layers 1 --max-tasks 8 --max-tests-per-loop 1`
- Force local LLM (Ollama):
  - `$Env:LLM_PROVIDER=\"ollama\"; $Env:OLLAMA_CHAT_TIMEOUT=\"120\"`

## Review and approval
- Inspect the latest JSONL suggestion file in `results/llm_suggestions/`.
- Approve/deny/edit items, then append approved items to the appropriate `LVL_{N}/tasklist.md`.
- L1 UI will add a page to streamline this flow (list → select → append).

## Troubleshooting
- Suggestions time out: increase `OLLAMA_CHAT_TIMEOUT`, switch to smaller model (e.g., `llama3.2:3b`), or use OpenAI.
- No actions recorded: ensure tasks exist in layer tasklists; expand safe scaffolds in the script; reduce pytest scope.
- DB or LLM not ready: run `scripts/health_probe.py` and fix reported components.

## Roadmap (High Priority)
- Suggestion reviewer CLI to generate a merge file of approvals.
- Script option to merge pre-approved suggestions into tasklists with provenance labels.
- More scaffolds for UI/docs so readiness flips faster.
- L1 Streamlit page for operating sessions and reviewing/approving suggestions.

