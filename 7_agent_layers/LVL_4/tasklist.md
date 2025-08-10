# Layer 4 – Agent Brain (Reasoning & Planning): Tasklist to move from 0 → 100

## Objectives
- Robust reasoning and planning producing executable, constrained plans

## KPI Targets
- **plan_success_rate** ≥ 90%; **replan_rate** < 15%
- **avg_plan_depth** 2–6 steps; **plan_validation_latency** ≤ 200 ms

---

## Phase 0 → 30: LLM Adapter (Week 1)
- [ ] Pluggable provider; prompt templates
- [ ] Safety filters for prompts/outputs
- [x] Tests: `test_llm_adapter.py`

Acceptance:
- [ ] Basic reasoning calls produce structured plan drafts

---

## Phase 30 → 60: Reasoning (Week 2)
- [ ] Multi‑step reasoning; evidence tracking
- [ ] Failure analysis and automatic plan revision
- [ ] Tests: `test_reasoning_multistep.py`

Acceptance:
- [ ] Revisions reduce failure rate; evidence attached

---

## Phase 60 → 80: Planning (Week 3)
- [ ] DAG plan schema; constraints (latency/cost/capabilities)
- [ ] Plan validator; interface to Layer 5
- [ ] Tests: `test_planning_dag.py`

Acceptance:
- [ ] Valid plans integrate with orchestrator; constraints enforced

---

## Phase 80 → 100: Optimization & Ops (Week 4)
- [ ] Plan metrics; replan triggers; docs/examples

Acceptance:
- [ ] KPIs achieved on system tests

---

## Chat Reasoning & Instruction Generation (Mesh: L4)
- [ ] Define `system_instructions` spec and prompt blocks for: overview, component-explain, how-to, and citation formatting
- [ ] Implement citation rendering: map metadata.path → relative repo link in answers
- [ ] Add answer quality guardrails: require grounded claims or explicitly say "unknown"
- [ ] Add follow-up suggestions generator (3 concise next questions)
- [ ] Tests: `test_chat_reasoning_quality.py`