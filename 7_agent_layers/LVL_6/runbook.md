# Runbook – Layer 6: Memory & Feedback

## Common symptoms
- Low recall@k; rapid store growth; consolidation lag

## Diagnostics
- Check recall@5; write_p95; growth_vs_consolidation metrics
- Inspect consolidation logs; verify TTL policies

## Remedies
- Enable HNSW; adjust dim/metric; tune consolidation cadence
- Improve write policy (dedup, usefulness thresholds)
