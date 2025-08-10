# Layer 1 – Human Interface: Tasklist to move from 0 → 100

## Objectives
- Conversational interface with reliable intent detection and UX
- Multi‑modal I/O (text, voice, images) with accessibility
- Structured feedback capture to inform Memory/Goals

## KPI Targets
- **intent_accuracy** ≥ 90%
- **ui_p95_latency** ≤ 300 ms
- **error_rate** < 1%; **feedback_capture_rate** ≥ 80%

---

## Phase 0 → 30: NL Interface (Week 1)
- [x] `src/layers/human_interface.py` scaffold (process_input)
- [ ] Simple intent classifier (rules + small LLM prompt)
- [ ] Connect to context retrieval (Layer 6) and tool orchestration (Layer 5)
- [x] Feedback endpoint (thumbs up/down); store to Memory
- [x] Tests: `test_human_interface_basic.py`

High Priority (MVP Monday):
- [ ] Streamlit Cloud UI shell (global navbar, auth stub)
- [ ] Main Chat page: system Q&A, start/stop self_dev_session, view reports, approve/deny/edit LLM suggestions
- [ ] Layers pages (L1–L7): description, history of completed tasks, current functionality review, approved tasklog, local mesh
- [ ] Global Mesh page: agent-wide mesh visualization + edit/troubleshoot guide
- [ ] Health page: DB/Ollama/GPU readiness, /performance metrics
- [ ] Tools/Resources/APIs page: MCP discovery list, endpoints, configs
- [ ] Backend API deployment via Docker Compose (API + Postgres + Ollama); Streamlit connects via env API_BASE

Acceptance:
- [ ] User text → intent + response; feedback persisted

---

## Phase 30 → 60: Multi‑modal (Week 2)
- [ ] Voice I/O (STT/TTS) behind feature flag
- [ ] Image/doc upload preview with safe rendering
- [ ] Accessibility (keyboard navigation, color contrast)
- [ ] Tests: `test_multimodal_smoke.py`

Acceptance:
- [ ] Voice/image flows work in dev; accessibility checks pass

---

## Phase 60 → 80: Conversation Management (Week 3)
- [x] Short‑term window + autosummary integration (Layer 6)
- [ ] Session state persistence; basic personalization
- [ ] Error recovery: clarify intent prompts
- [ ] Tests: `test_conversation_flow.py`

Acceptance:
- [ ] Multi‑turn flows maintain context; autosummary reduces tokens

---

## Phase 80 → 100: Advanced UX (Week 4)
- [ ] Persona toggles; adaptive UI components
- [ ] Inline trace viewer (compact chain summary)
- [ ] UX metrics dashboard (p95 latency, error rates)
- [ ] Docs and examples

Acceptance:
- [ ] UX KPIs achieved; trace viewer shows per‑turn context

---

## Deliverables
- `human_interface.py`, feedback API, tests, docs

## Risks & Mitigations
- Ambiguous intents → clarify prompts + rules fallback
- Token costs → autosummary + minimal context window

## Ownership & Timeboxes
- W1: NL interface + feedback
- W2: Multi‑modal
- W3: Conversation manager
- W4: Advanced UX & metrics