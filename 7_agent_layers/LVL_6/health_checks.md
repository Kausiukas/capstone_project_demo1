# Health Checks – Layer 6: Memory & Feedback

## Probes
- query_context("status") returns snippets; store_interaction writes

## Thresholds
- recall@5 ≥ 0.6; write_p95 ≤ 150 ms

## Commands
- python scripts/run_traces.py --dim 256 --summarize
