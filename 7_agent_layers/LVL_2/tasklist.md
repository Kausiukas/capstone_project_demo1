# Layer 2 – Information Gathering & Context: Tasklist to move from 0 → 100

## Objectives
- Comprehensive, fresh, validated context for reasoning and tools
- Low-latency collectors and concise synthesis briefs

## KPI Targets
- **freshness_mean** < 6h (configurable); **coverage** ≥ 90% of targets
- **collector_p95** ≤ 500 ms; **error_rate** < 2%

---

## Phase 0 → 30: Collectors (Week 1)
- [x] HTTP fetch collector with content sniffing + size caps
- [x] Filesystem watcher (optional for dev)
- [x] System metrics/log taps
- [x] Tests: `test_collectors_basic.py`

Acceptance:
- [ ] Collectors fetch and store snippets with tags and provenance

---

## Phase 30 → 60: Normalization & Synthesis (Week 2)
- [ ] Type detection; safe parsing; preview extraction
- [ ] Synthesis briefs (<1 KB) per topic/query
- [x] Tests: `test_normalize_summarize.py`

Acceptance:
- [ ] Multi-source snippets merged into concise briefs with provenance

---

## Phase 60 → 80: Scheduling & Validation (Week 3)
- [ ] Periodic collection loop with backoff and robots.txt respect
- [ ] Freshness scoring; confidence metrics
- [ ] Tests: `test_schedule_validation.py`

Acceptance:
- [ ] Scheduled runs meet freshness targets; validation metrics emitted

---

## Phase 80 → 100: Performance & Ops (Week 4)
- [ ] Parallel fetch pool with rate limits per domain
- [ ] Dashboards for freshness/coverage/error/p95
- [ ] Docs and examples

Acceptance:
- [ ] KPIs achieved on local benchmarks; dashboards reflect state