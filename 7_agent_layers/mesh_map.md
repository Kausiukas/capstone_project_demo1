# Agent Mesh Map (one-line-per-layer)

```yaml
version: 1.1
last_updated: 2025-08-08
```

| Layer | Inputs | Outputs | Key APIs | Data/Stores | Depends On | Health |
|---|---|---|---|---|---|---|
| 1 Human Interface | user_text; media | intents; ui_feedback | process_input; feedback.post | session_state | 2,5,6 | intent_accuracy; ui_p95_latency; error_rate |
| 1.1 MVP UI (High-priority) | chat_actions; approvals | approved_tasks; session_cmds | streamlit.chat; sessions.run; suggestions.review | ui_state | 5,6,7 | page_load_ms; action_success |
| 2 Info & Context | files; http; system | context_snippets; briefs | gather_context; preview_file; fetch | index_cache | 6 | freshness; coverage; fetch_p95 |
| 3 Goals & Behaviors | intents; prefs; history | prioritized_goals; constraints | goal_mgmt.define/prioritize; policies | goals_store | 6,4 | goal_completion; conflicts |
| 4 Agent Brain | intents; context; goals | plans(DAG); decisions | reason_and_plan; revise_plan | reasoning_traces | 2,3,5 | plan_success; replan_rate; avg_plan_depth |
| 5 Tools & API | plans/steps | tool_results; exec_traces | orchestrate_task; /tools/orchestrate; /tools/registry/health; /api/v1/tools/call | exec_logs; tool_metrics | MCP,6 | chain_success; step_p95; retries; fallback_rate |
| 5.1 Self-Dev Orchestrator | dashboard; llm; approvals | suggestions.jsonl; session_reports | self_dev_session.run; tasklog_dashboard.auto | results_dir | 4,6,7 | cycles; actions; llm_tokens |
| 6 Memory & Feedback | interactions; embeddings; feedback | retrieved_context | store_interaction; query_context; feedback.apply | pgvector_store | DB | recall@5; write_p95; growth_vs_consolidation |
| 7 Infra & Security | traffic; secrets | telemetry; enforcement | /health; /metrics; /performance/* | logs; metrics; backups | cloud/docker | uptime; error_rate; saturation |

## Edges (producers → consumers)
```yaml
- 1:intents -> 3,4
- 2:context_snippets -> 4,1,5
- 3:prioritized_goals -> 4
- 4:plans -> 5
- 5:exec_traces -> 6
- 6:retrieved_context -> 1,4,5
- 7:telemetry -> 1,2,3,4,5,6

## Deployment notes (MVP)
- Frontend: Streamlit Cloud app (pages: Chat/Sessions/Reports/Suggestions/Layers/Mesh/Health/Tools) consuming API_BASE
- Backend: FastAPI (`src/mcp_server_enhanced_tools.py`) served via Docker Compose alongside Postgres+pgvector and Ollama
```

## Active Mesh Tasks (Chat improvements)
- L4 (Reasoning): system_instructions spec; citation rendering; grounded-answer guardrails; follow-up suggestions; tests
- L6 (Memory): Ollama embeddings; chunking+tags; feedback-weighted ranking; incremental indexer; tests

## File structure & authoring guidelines
- Directory layout per layer: `7_agent_layers/LVL_{1..7}/`
  - `layer{n}.md`: Purpose → Core Components → Status (YAML) → Implemented → Missing → Implementation Plan (Phase 1..3 with {#phase-1} anchors) → Success Metrics → References
  - `mesh.md`: Expanded mesh (profile, health/SLIs, failure modes, alerts, edges)
  - `tasklist.md`: Phase 0→30, 30→60, 60→80, 80→100 with acceptance criteria and KPIs
- Keep each doc ≤ 2,000 tokens; avoid duplicated prose; link to related docs via relative paths
- Put runnable code in `snippets/`, not in docs; include minimal fenced stubs only
- Use consistent tables (one line per layer in this file); semicolon-delimited cell lists for token efficiency
- Include YAML headers for machine consumption where appropriate
