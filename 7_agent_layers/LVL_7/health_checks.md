# Health Checks – Layer 7: Infrastructure, Scaling & Security

## Probes
- /health 200 OK; /metrics responds; /performance/* returns data

## Thresholds
- uptime ≥ 99.9%; error_rate < 1%

## Commands
- curl http://127.0.0.1:8000/health
