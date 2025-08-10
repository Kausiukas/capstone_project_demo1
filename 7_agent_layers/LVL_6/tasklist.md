# Layer 6 – Memory & Feedback: Tasklist to move from 0 → 100

## Objectives
- High‑quality recall with compact storage; feedback‑driven ranking

## KPI Targets
- **recall@5** ≥ 0.6; **write_p95** ≤ 150 ms
- **duplication_ratio** trending ↓; **consolidation_lag** < 24h

---

## Phase 0 → 30: Short‑term + Ranking (Week 1)
- [x] Rolling window + autosummary compression
- [x] Feedback‑weighted ranking in query
- [x] Tests: `test_memory_window_ranking.py`

Acceptance:
- [ ] Window reduces tokens; ranking reflects feedback

---

## Phase 30 → 60: Consolidation (Week 2)
- [ ] Nightly cluster→summarize; archive originals; TTL
- [ ] PII redaction; purge by tag/user
- [ ] Tests: `test_consolidation_jobs.py`

Acceptance:
- [ ] Duplication down; recall stable/improved

---

## Phase 60 → 80: Performance (Week 3)
- [ ] HNSW index; metric toggle; retrieval optimizer
- [ ] Tests: `test_retrieval_performance.py`

Acceptance:
- [ ] recall@5 meets target with lower latency

---

## Phase 80 → 100: Ops & Metrics (Week 4)
- [ ] Dashboards and alerts (recall, growth, consolidation lag)
- [ ] Docs and examples

Acceptance:
- [ ] KPIs achieved; alerts configured

---

## Chat Memory & RAG Store (Mesh: L6)
- [ ] Replace hash embeddings with Ollama embeddings API (fallback to hash) and store embedding model/version
- [ ] Add source chunking with overlap and per-file tags; persist `path`, `section`, `layer`
- [ ] Implement feedback write-back: thumbs-up/down affects ranking weight
- [ ] Add nightly incremental indexer for `7_agent_layers/` diffs; deduplicate by content_hash
- [ ] Tests: `test_rag_store_quality.py`