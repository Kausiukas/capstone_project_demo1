# Health Checks – Layer 1: Human Interface

## Probes
- process_input("ping") → intent="generic_request" (stub)
- Feedback POST → stored in Memory

## Thresholds
- ui_p95_latency ≤ 300 ms
- error_rate < 1%

## Commands
- python -c "from src.layers.human_interface import HumanInterface as H;print(H().process_input('hello'))"
