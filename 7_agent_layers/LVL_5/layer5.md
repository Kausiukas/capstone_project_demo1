# 🔧 Layer 5: Tools & API Layer

## Function & Purpose
This layer provides a unified capability surface to discover, normalize, select, and execute tools (local and across MCP servers) with reliability and observability. It reduces burden on the Brain (Layer 4) by handling multi-step orchestration and on Memory (Layer 6) by emitting concise, structured execution traces instead of ad‑hoc context dumps.

### Goals
- Convert intents into executable chains with typed inputs/outputs
- Run tools safely (validation, timeouts, retries, backoff, fallbacks)
- Support parallel/conditional flows and cross-server data passing
- Record parsable traces for learning and auditing
- Continuously improve selection policies using usage/feedback signals

## Core Components
- **Tool Registry**: Dynamic discovery and normalization (ToolSpec), health/latency/cost metadata
- **Tool Orchestrator**: Intent→plan, chain/DAG builder, dependency management, fallbacks
- **API Integrations**: Third‑party connectors, key management, rate limiting, health checks
- **Execution Engine**: Validation, adapters, retries/backoff, timeouts, tracing
- **Tool Learning**: Usage analytics, success/failure attribution, policy updates

## Current Implementation Status ✅ 90% Complete

### ✅ What's Implemented
- 5 core MCP tools (ping, list_files, read_file, get_system_status, analyze_code)
- Universal file access; content preview/analysis
- Performance monitoring and basic API authentication
- Baseline error handling

### ❌ What's Missing
- Multi‑server tool discovery and registration
- Declarative tool composition and chaining (DAG)
- External API integrations (with rate limits/backoff)
- Tool selection optimization (latency/success/cost)
- Advanced orchestration (parallel/conditional, aggregation)

## Implementation Plan

### Phase 1: Tool Orchestrator (Week 1–2)
Deliverables:
- Tool Registry: discover from N MCP bases (`/tools/list`), normalize to `ToolSpec`
- Rule‑first Planner: task → ordered chain with simple dependencies
- Executor: sequential execution, validation, timeouts, single retry, JSON trace → Memory

```python
# New component: src/layers/tool_orchestrator.py
class ToolOrchestrator:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.execution_engine = ExecutionEngine()
        self.learning_engine = ToolLearningEngine()
    
    def orchestrate_task(self, task_description):
        """Intelligently orchestrate tools for complex tasks"""
        tools_needed = self.analyze_task_requirements(task_description)
        execution_plan = self.create_tool_chain(tools_needed)
        result = self.execute_plan(execution_plan)
        self.learn_from_execution(execution_plan, result)
        return result
    
    def create_tool_chain(self, tools_needed):
        """Create optimal tool execution chain"""
        chain = []
        for tool in tools_needed:
            dependencies = self.identify_dependencies(tool)
            chain.append({
                'tool': tool,
                'dependencies': dependencies,
                'execution_order': len(chain)
            })
        return chain
```

### Phase 2: External API Integration (Week 3–4)
- Add at least 2 connectors (e.g., HTTP fetch with API key, GitHub API)
- Env‑based key management; per‑tool rate limits & backoff; error taxonomy
- Health probes and SLAs exposed in registry; simple budget guards

### Phase 3: Advanced Orchestration (Week 5–6)
- DAG execution (parallel branches, conditional steps, aggregation)
- Selection policy: score candidates (latency/success/cost) + learned priors
- Result synthesizer: concise aggregated outputs (<2KB) + artifact links

## Tasklog
- 2025‑08‑08: Defined Orchestrator MVP scope (registry, planner, executor, tracing)
- 2025‑08‑08: Identified selection/rate‑limit/backoff policies; error taxonomy draft
- 2025‑08‑08: Planned multi‑server discovery via `/tools/list` → normalized `ToolSpec`
- Next: Implement `src/layers/tool_orchestrator.py`; add smoke test `tests/phase1/test_orchestrator_smoke.py`; wire traces to Memory
