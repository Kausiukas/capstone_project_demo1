# Layer 6 Mesh – Memory & Feedback

```yaml
version: 1.0
last_updated: 2025-08-08
```

## Profile
- Inputs: interactions; embeddings; feedback signals
- Outputs: retrieved_context; memory_stats; feedback_updates
- Key APIs/Funcs: store_interaction; query_context; apply_feedback
- Internal Services: ShortTermWindow; LongTermStore; Consolidator
- External Services: pgvector DB; Embedding provider; Orchestrator traces
- Data/Stores: pgvector_store; summaries; indexes
- Depends On: DB; Layers 1,5

## Health/SLIs
- recall@k; write latency; store growth vs consolidation

## Failure Modes
- dimension mismatch; duplicate bloat; stale recalls

## Alerts
- low recall@k; rapid growth; consolidation lag

## Edges
- Produces: retrieved_context -> Layers 1,4,5; feedback_updates -> Layers 3,5
- Consumes: exec_traces <- Layer 5; interactions <- Layer 1
