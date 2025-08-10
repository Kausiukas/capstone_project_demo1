# Layer 7 – Infrastructure, Scaling & Security: Tasklist to move from 0 → 100

## Objectives
- Secure, reliable, and scalable operation for the agent stack

## KPI Targets
- **uptime** ≥ 99.9%; **error_rate** < 1%
- **p95_latency** within SLOs; **backup_success** = 100%

---

## Phase 0 → 30: Security (Week 1)
- [ ] Rate limiting; stricter CORS; API key rotation
- [ ] Security headers and input validation
- [ ] Tests: `test_security_basics.py`

Acceptance:
- [ ] Requests governed by limits; headers present; inputs validated

---

## Phase 30 → 60: Scaling (Week 2)
- [ ] Horizontal scaling plan; health‑based routing
- [ ] Resource budgets; autoscaling hooks
- [ ] Tests: `test_scaling_health.py`

Acceptance:
- [ ] Load scenarios pass SLOs; budgets enforced

---

## Phase 60 → 80: Observability (Week 3)
- [ ] Unified metrics/logs; error budgets; alerts
- [ ] Perf dashboards; runbooks
- [ ] Tests: `test_observability.py`

Acceptance:
- [ ] Alerts actionable; dashboards reflect SLOs

---

## Phase 80 → 100: Ops Hardening (Week 4)
- [ ] Backups/DR drills; image patching cadence; dependency scanning
- [ ] Docs and procedures

Acceptance:
- [ ] Backups/restores verified; scans clean; docs complete
