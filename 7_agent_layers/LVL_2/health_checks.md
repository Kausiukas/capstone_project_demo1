# Health Checks – Layer 2: Information Gathering & Context

## Probes
- gather_context("status") returns snippets
- preview_file(path) returns safe preview

## Thresholds
- freshness_mean < 6h; fetch_p95 ≤ 500 ms

## Commands
- python scripts/index_local_resources.py --roots src docs --dim 256 --limit 10
