# Runbook – Layer 1: Human Interface

## Common symptoms
- High UI latency; intents not detected; feedback not saved

## Diagnostics
- Check traces: tail logs/agent_traces.jsonl
- Verify intent_accuracy and ui_p95_latency in mesh metrics
- Exercise API: process_input with sample text

## Remedies
- Reduce context window; enable autosummary (Layer 6)
- Improve rules/LLM prompt for intent extraction
- Ensure feedback endpoint writes to Memory
