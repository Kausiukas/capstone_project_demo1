# Layer 4 Mesh – Agent Brain (Reasoning & Planning)

```yaml
version: 1.0
last_updated: 2025-08-08
```

## Profile
- Inputs: intents; context_snippets; prioritized_goals
- Outputs: plans(DAG); decisions; reasoning_summaries
- Key APIs/Funcs: reason_and_plan(ctx, goals); revise_plan()
- Internal Services: ReasoningEngine; PlanningEngine; DecisionEngine
- External Services: ToolOrchestrator; MemorySystem
- Data/Stores: reasoning_traces; plan_cache
- Depends On: Layers 2,3,5,6

## Health/SLIs
- plan success; replan rate; average plan depth

## Failure Modes
- hallucinated steps; infeasible plans; context drift

## Alerts
- replan spikes; low plan success for intents

## Edges
- Produces: plans -> Layer 5; reasoning_summaries -> Layer 6
- Consumes: context_snippets <- Layer 2; goals <- Layer 3; retrieved_context <- Layer 6
