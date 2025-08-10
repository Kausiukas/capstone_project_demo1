# Directory Requirements Template (per layer)

This template lists the minimal and recommended files each layer should contain, aligned with the mesh map and tasklists.

## Mesh alignment
- Each layer must expose: Inputs, Outputs, Key APIs/Functions, Data/Stores, Dependencies, Health/SLIs.
- Emit telemetry events (stage_start/end) with trace_id and metrics defined in the mesh.

---

## Base structure per layer (LVL_{1..7}/)
- `layer{n}.md` (required)
  - Purpose → Core Components → Status (YAML) → Implemented → Missing → Implementation Plan (Phase 1..3 {#phase-1}) → Success Metrics → References
- `mesh.md` (required)
  - Profile (inputs/outputs/APIs/services/stores/deps), Health/SLIs, Failure modes, Alerts, Edges
- `tasklist.md` (required)
  - Phases 0→30, 30→60, 60→80, 80→100 with acceptance criteria and KPIs
- `runbook.md` (recommended)
  - Symptoms → Diagnostics → Remedies; quick commands
- `health_checks.md` (recommended)
  - Probes, expected outputs, thresholds
- `playbooks.md` (recommended)
  - Incident recipes (rate limits, DB contention, planner loops)
- `decisions.md` (recommended)
  - Key design choices and trade‑offs
- `changelog.md` (recommended)
  - Date, change, impact, rollback
- `snippets/` (optional)
  - Runnable examples; keep code out of docs if not necessary
- `tests/` (optional, or add to central tests/*)
  - Focused unit/integration tests for this layer

---

## Layer‑specific additions

### LVL_1 (Human Interface)
- APIs/Funcs: `process_input(text)`, `feedback.post()`
- Telemetry metrics: intent_accuracy, ui_p95_latency, error_rate, feedback_rate
- Files:
  - `snippets/human_interface.py` (minimal stub for process_input)
  - `tests/test_human_interface_basic.py`

### LVL_2 (Information Gathering & Context)
- APIs/Funcs: `gather_context(query)`, `preview_file(path)`, `fetch(url)`
- Telemetry metrics: freshness, coverage, fetch_p95
- Files:
  - `collectors/` (http.py, fs.py), `synthesizer.py`, `validators.py`
  - `tests/test_collectors_basic.py`, `test_normalize_summarize.py`

### LVL_3 (Structure, Goals & Behaviors)
- APIs/Funcs: `goal_mgmt.define/prioritize`, `policies()`
- Telemetry metrics: goal_completion_rate, conflicts
- Files:
  - `goal_tracker.py`, `priority_manager.py`, `ethics.py`, `personality.py`
  - `tests/test_goals_foundation.py`, `test_traits_learning_goals.py`

### LVL_4 (Agent Brain)
- APIs/Funcs: `reason_and_plan(ctx, goals)`, `revise_plan()`
- Telemetry metrics: plan_success, replan_rate, avg_plan_depth
- Files:
  - `llm_adapter.py`, `reasoning_engine.py`, `planning_engine.py`, `decision_engine.py`
  - `tests/test_llm_adapter.py`, `test_reasoning_multistep.py`, `test_planning_dag.py`

### LVL_5 (Tools & API)
- APIs/Funcs: `orchestrate_task`, `/tools/orchestrate`, `/tools/registry/health`, `/api/v1/tools/call`
- Telemetry metrics: chain_success, step_p95, retries, fallback_rate
- Files:
  - `tool_orchestrator.py` (ToolSpec, ToolRegistry, ToolPlanner, ExecutionEngine, Orchestrator)
  - `errors.py` (taxonomy)
  - `config/orchestrator.yaml`
  - `tests/test_tool_registry.py`, `test_planner_rules.py`, `test_executor_sequential.py`, `test_rate_limit_backoff.py`, `test_tool_selection.py`, `test_executor_dag.py`

### LVL_6 (Memory & Feedback)
- APIs/Funcs: `store_interaction`, `query_context`, `feedback.apply`
- Telemetry metrics: recall@5, write_p95, growth_vs_consolidation
- Files:
  - `consolidation_job.py`, `ranking.py`, `window.py`
  - `config/memory.yaml` (dim, HNSW, TTL, cadence)
  - `tests/test_memory_window_ranking.py`, `test_consolidation_jobs.py`, `test_retrieval_performance.py`

### LVL_7 (Infrastructure, Scaling & Security)
- APIs/Funcs: `/health`, `/metrics`, `/performance/*`
- Telemetry metrics: uptime, error_rate, saturation
- Files:
  - `security.py` (rate_limit, CORS, validation), `observability.py`
  - Runbooks for DR, backups; `tests/test_security_basics.py`, `test_observability.py`

---

## Authoring rules
- Keep each document ≤ 2,000 tokens; prefer tables and bullets
- Use consistent headings and YAML status blocks for machine parsing
- Reference other docs via relative links; avoid prose duplication across layers
- Emit telemetry at stage boundaries; attach layer‑specific metrics from mesh
- Put production code in `src/` and tests in `tests/`; docs in `7_agent_layers/`
