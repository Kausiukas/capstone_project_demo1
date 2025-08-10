# Layer 5 – Tools & API: Tasklist to move from 0 → 100

## Objectives
- Unified tool capability layer (local + MCP servers)
- Reliable orchestration (plan → execute), observability, learning
- Compact outputs and structured traces to support Memory/Goals

## KPI Targets
- **chain_success_rate** ≥ 95%
- **step_p95** ≤ 800 ms (local dev), **retries_used** < 10%
- **fallback_rate** < 10%, **schema_mismatch** = 0
- **trace_size_mean** < 2 KB; full coverage of target MCP servers

---

## Phase 0 → 30: Scaffolding & Registry (Week 1)
- [x] Create `src/layers/tool_orchestrator.py` with:
  - [ ] `ToolSpec` (name, server, inputs, outputs, auth, limits)
  - [ ] `ToolRegistry` (discover/cache/health)
- [ ] Discovery (single MCP): `GET {mcp}/tools/list` → normalize to `ToolSpec`
- [ ] TTL cache; health (last_seen, 5xx/429 counters)
- [x] Config: `config/orchestrator.yaml` (MCP endpoints, timeouts, limits, retries)
- [x] Tests: `tests/phase1/test_tool_registry.py` (mocks)

Acceptance:
- [ ] Registry returns normalized ToolSpec list (≥ 5 tools); TTL respected; health populated

---

## Phase 30 → 60: Planner + Sequential Executor (Week 2)
- [ ] `ToolPlanner` (rules-first): intent → ordered steps with simple dependencies
- [ ] IO schema mini-DSL (json fields) + validators
- [ ] `ExecutionEngine.run_chain(steps, payload)`:
  - [ ] Input validation per step
  - [ ] Per-step timeout; 1 retry with backoff
  - [ ] Standard error taxonomy (rate_limit|timeout|schema|server_error|unavailable)
  - [ ] Fallback to alternate server on 429/5xx
- [ ] Orchestrator: `ToolOrchestrator.orchestrate_task(text, context)`
- [ ] Emit traces (JSON) via existing harness with step metrics
- [ ] Endpoint (optional): `POST /tools/orchestrate`
- [x] Tests: `test_planner_rules.py`, `test_executor_sequential.py`, `test_error_mapping.py`, `test_orchestrator_smoke.py`

Acceptance:
- [ ] Smoke task “analyze file then summarize” passes end-to-end with trace; retries/fallback verified; no schema mismatch

---

## Phase 60 → 80: Multi‑server + Policies + Adapters (Week 3)
- [ ] Registry: multi-MCP discovery; per-server health/backoff state
- [ ] Selection policy: score (success_rate, p95_latency, recent_429/5xx, cost_hint)
- [ ] Per-tool/server rate limiting (token bucket) + exponential backoff
- [ ] Cross-server adapters for common IO shapes (text/path/url/code)
- [ ] Aggregator node to combine step outputs (compact synthesis)
- [ ] Metrics: chain_success_rate, step_p95, retries, fallback_rate, selection_decisions
- [x] Health endpoint: `GET /tools/registry/health`
- [x] Tests: `test_tool_selection.py`, `test_rate_limit_backoff.py`, `test_io_adapters.py`, `test_orchestrator_metrics.py`

Acceptance:
- [ ] Two MCP bases discovered; selection prefers best by score; health endpoint returns per-server/tool stats

---

## Phase 80 → 100: DAG + Synthesis + Ops (Week 4)
- [ ] DAG planner: parallel branches and conditional edges
- [ ] `ExecutionEngine.run_dag()` with small worker pool
- [ ] Result synthesizer: concise (< 2 KB) outputs + artifact links
- [ ] Budget guards (latency/cost); early abort on budget breach
- [ ] Dashboard (basic): orchestrator KPIs (reuse perf UI if available)
- [ ] Docs: `docs/layers/layer5_tools_api.md` (Purpose → Components → Status → Plan → Metrics)
- [x] Tests: `test_executor_dag.py` (parallel + join), budget tests

Acceptance:
- [ ] Parallel chain with conditional step runs green; budgets enforced; KPIs hit on local benchmarks

---

## Deliverables Checklist
- [ ] `tool_orchestrator.py` (Registry, Planner, Executor, Orchestrator, ToolSpec)
- [ ] `errors.py` (taxonomy and mapping)
- [ ] `orchestrator.yaml` (config)
- [ ] Health endpoint(s) + metrics emission
- [ ] Full unit/integration test set green
- [ ] Docs + examples

## 5.1 Self-Development Orchestrator (High Priority)
- [ ] `self_dev_session.md` (docs: purpose, scope, flow, usage)
- [ ] Suggestions review gate: CLI helper to list/approve/deny/edit suggestions → append to layer tasklists
- [ ] Extend scaffolds for UI/docs/tests tasks across layers; record flips as actions
- [ ] Flags: bias-to-layer (e.g., L1), max tasks per layer, merge approved suggestions file
- [ ] Emit richer counters (approved/denied/edited), and attach suggestion provenance to tasklist entries
- [ ] Page in L1 UI to operate the self-development loop and review results

## Risks & Mitigations
- Heterogeneous tool schemas → normalize + adapters; strict validators
- Rate-limit storms → token bucket + backoff + fallback rotation
- Silent failures → standard error shape + structured traces
- Performance regressions → p95 monitoring + selection feedback

## Ownership & Timeboxes
- Week 1: Registry scaffolding + tests
- Week 2: Planner + sequential executor + smoke
- Week 3: Policies, adapters, health + metrics
- Week 4: DAG, synthesis, budgets, docs