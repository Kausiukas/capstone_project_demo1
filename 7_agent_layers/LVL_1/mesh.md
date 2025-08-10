# Layer 1 Mesh – Human Interface

```yaml
version: 1.0
last_updated: 2025-08-08
```

## Profile
- Inputs: user_text; media (images/docs); ui_events
- Outputs: intents; ui_feedback; conversation_turns
- Key APIs/Funcs: process_input(text); render_preview(path); post_feedback(record_id, up|down)
- Internal Services: NLPEngine; ConversationManager; ResponseGenerator
- External Services: MemorySystem (store/query); ToolOrchestrator (plan/exec); Embedding provider
- Data/Stores: session_state; ui_metrics
- Depends On: Layer 2, 5, 6

## Health/SLIs
- p50/p95 UI latency; input error rate; feedback submission success

## Failure Modes
- NLP unavailable; embedding errors; oversized payloads; file preview timeouts

## Alerts
- UI latency > threshold; feedback drop rate; preview error spikes

## Edges
- Produces: intents -> Layers 3,4; ui_feedback -> Layer 6
- Consumes: retrieved_context <- Layer 6; tool_results <- Layer 5
