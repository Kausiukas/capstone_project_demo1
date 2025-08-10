# Layer 2 Mesh – Information Gathering & Context

```yaml
version: 1.0
last_updated: 2025-08-08
```

## Profile
- Inputs: file_paths; http_urls; system_signals
- Outputs: context_snippets; briefs; provenance
- Key APIs/Funcs: gather_context(query); preview_file(path); fetch(url)
- Internal Services: Collectors; Normalizers; Synthesizer; Validators
- External Services: MemorySystem (index/query)
- Data/Stores: index_cache; fetch_logs
- Depends On: Layer 6

## Health/SLIs
- freshness; coverage; error rate; fetch P95

## Failure Modes
- fetch blocked; parse errors; stale cache

## Alerts
- low freshness; high fetch failures; abnormal content types

## Edges
- Produces: context_snippets -> Layers 1,4,5
- Consumes: retrieved_context <- Layer 6
