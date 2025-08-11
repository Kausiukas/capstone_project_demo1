# Ping Trace Plan — 2025-08-11

## Objective
Design and iterate a tracing-capable ping so the UI can reliably retrieve an API identifier together with `pong`, both locally and via a Cloudflare tunnel.

## Target Contract (minimal)
POST `/api/v1/tools/call` body `{ "name": "ping", "arguments": {} }`
Response (200 JSON):

```
{
  "content": [{"type":"text","text":"pong"}],
  "instance": {
    "hostname": "capstone_api",
    "started_at": 1723380000,
    "image_tag": "kausiukas/langflow-connect-api:latest",
    "commit_sha": "<short-sha>",
    "build_time": "2025-08-11T10:00:00Z"
  }
}
```

Acceptance: `instance` must be present and non-empty.

## Iteration Plan
1. Backend capability (done)
   - Return `instance` metadata in ping response.
   - Mirror the same metadata in `/health.instance` for fallback checks.
2. UI surfacing (done)
   - Tools/APIs → "Ping API instance" shows `pong` and the `instance` JSON.
3. Fallback path (verify)
   - If ping lacks `instance`, call `/health` and display `.instance`.
4. Version/identity hardening (next)
   - Add response headers to all requests: `X-Api-Hostname`, `X-Api-Started-At`, `X-Api-Image-Tag`, `X-Api-Commit`.
   - UI shows these headers when present.
5. Instance stamp (optional)
   - Add `instance_id = sha1(hostname|started_at|commit_sha)` to ping and health for compact display.
6. Trace timing (optional)
   - Include `received_at`, `server_now`, and `latency_ms` in ping for quick network/tunnel diagnostics.

## Test Matrix
- Environments
  - Localhost direct: `http://localhost:8000`
  - Tunnel: `https://<quick>.trycloudflare.com`
- Scenarios
  1) Ping with correct `X-API-Key` → expect `pong` + non-empty `instance`.
  2) Ping missing key → `401`.
  3) `/health` with key → expect `.instance`.
  4) Tunnel freshness: after rebuild, confirm `.instance.started_at` > previous.
- PowerShell
  - `$hdr=@{ 'X-API-Key'='demo_key_123' }`
  - `Invoke-RestMethod "$API/api/v1/tools/call" -Headers $hdr -Method Post -ContentType 'application/json' -Body '{"name":"ping","arguments":{}}'`
  - `Invoke-RestMethod "$API/health" -Headers $hdr`
- curl
  - `curl -sS -H "X-API-Key: demo_key_123" -H "Content-Type: application/json" -d '{"name":"ping","arguments":{}}' $API/api/v1/tools/call`

## Troubleshooting Steps
- 404 on ping or empty `instance`: tunnel points to stale container. Rebuild with `--no-cache`, recreate container, restart tunnel.
- `Invalid API key`: UI/browser calls omit header. Only test via UI buttons or include header.
- No LLM/RAG info: that is separate; use `/status/components` for a broader check once tunnel serves the new build.

## Definition of Done
- UI Ping shows `pong` and a non-empty `instance` both on localhost and via the tunnel.
- `/health.instance` mirrors the same identity.
- Optional: response headers with identity available for any endpoint.

## Next Actions
- Implement identity headers middleware (backend) and display in UI Ping panel.
- Add compact `instance_id` hash to ping/health for quick comparison across environments.


## Smart Ping: Instance Identification Without Server Cooperation

### Goal
Fingerprint the currently connected API instance using network- and protocol-level signals even when the server does not explicitly expose identity in its payloads.

### Signals to Collect
- DNS/IP
  - Resolve hostname to IPs (A/AAAA); note changes and ordering.
  - Reverse DNS of the connected IP.
- TCP/TLS
  - Connection latency (SYN→ACK), TLS handshake time.
  - TLS version (e.g., TLS 1.3), ALPN (h2 vs http/1.1), cipher suite, curve.
  - Certificate SHA-256 fingerprint, issuer, SANs, notBefore/notAfter.
- HTTP/1.1 or HTTP/2
  - Status, Server, Date, Via, Alt-Svc, ETag, Cache-Control, Location (redirects).
  - Unique IDs if present: `x-request-id`, `traceparent`, `cf-ray`, `x-envoy-upstream-service-time`.
- Cloudflare-specific (if tunneled)
  - `cf-ray`, `cf-cache-status`, `server: cloudflare`, `cf-visitor`.
  - POP/colo inference from `cf-ray` suffix.
- Timing
  - `ttfb_ms`, `download_ms`, total latency; `Date` header skew vs local clock.

### Fingerprint Recipe (deterministic)
Build a stable signature string from mostly invariant signals:
`signature_material = host + '|' + primary_ip + '|' + tls_cert_sha256 + '|' + tls_alpn + '|' + http_server_header`
Then compute `signature = sha256(signature_material)`.

Recommended JSON payload:
```
{
  "target_url": "https://example.trycloudflare.com",
  "resolved_ips": ["203.0.113.10"],
  "tcp_connect_ms": 42,
  "tls": {
    "version": "TLS1.3",
    "alpn": "h2",
    "cipher": "TLS_AES_128_GCM_SHA256",
    "cert_sha256": "...",
    "issuer": "...",
    "san": ["example.trycloudflare.com"],
    "not_before": "...",
    "not_after": "..."
  },
  "http": {
    "status": 200,
    "server": "cloudflare",
    "date": "...",
    "via": null,
    "cf_ray": "..."
  },
  "timing_ms": {"ttfb": 120, "download": 8, "total": 128},
  "signature": "sha256:..."
}
```

### PowerShell Probes (Windows-friendly)
- Resolve host/IPs
  - `Resolve-DnsName $HOST`
  - `[System.Net.Dns]::GetHostAddresses($HOST) | ForEach-Object { $_.IPAddressToString }`
- HTTP headers (no body)
  - `$hdr=@{ 'X-API-Key'='demo_key_123' }`
  - `(Invoke-WebRequest -Uri $API -Headers $hdr -Method Head -UseBasicParsing).Headers`
- HTTP GET with timing (approx)
  - `$sw=[System.Diagnostics.Stopwatch]::StartNew(); $r=Invoke-WebRequest -Uri "$API/health" -Headers $hdr -UseBasicParsing; $sw.Stop(); @{ status=$r.StatusCode; server=$r.Headers['Server']; date=$r.Headers['Date']; ms=$sw.ElapsedMilliseconds } | ConvertTo-Json`
- TLS certificate fingerprint (port 443)
  - PowerShell snippet:
```
$hostName = ($API -replace '^https?://','').TrimEnd('/')
if ($hostName -match '([^/:]+)(:(\d+))?') { $hostName = $Matches[1] }
$client = New-Object System.Net.Sockets.TcpClient($hostName,443)
$ssl = New-Object System.Net.Security.SslStream($client.GetStream(),$false,({$true}));
$ssl.AuthenticateAsClient($hostName)
$cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($ssl.RemoteCertificate)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
$hash = $sha256.ComputeHash($cert.RawData)
$fp = -join ($hash | ForEach-Object { $_.ToString('x2') })
@{ cert_sha256=$fp; issuer=$cert.Issuer; subject=$cert.Subject; not_after=$cert.NotAfter } | ConvertTo-Json
$ssl.Dispose(); $client.Close()
```

### Minimal Python CLI (optional for accuracy)
Create `scripts/smart_ping.py` later; it will:
- Resolve DNS
- Measure TCP connect and TLS handshake
- Capture TLS details with `ssl` (`ALPN`, cipher, certificate)
- Fetch `HEAD` and `GET /health` headers with `httpx`
- Emit the JSON payload and the `signature`

### Test Steps
1. Collect fingerprints for localhost and tunnel; store to `results/ping_trace/*.json`.
2. Change backend version (rebuild); confirm cert/IP/headers or `signature` reflect the new instance or route.
3. During incidents (empty instance in ping), rely on `signature` deltas to spot stale tunnel or wrong target.

### UI Integration (follow-up)
- Add a Smart Ping tab in `web/streamlit_app.py` showing:
  - Resolved IPs, TLS cert fingerprint, ALPN/version, Cloudflare headers (if any), and `signature`.
  - Copy-to-clipboard for the fingerprint JSON.

### Egress (Tunnel → Origin) Detection
- Decision signals
  - Reached Cloudflare edge: presence of `server: cloudflare` and a valid CF certificate.
  - Origin reachable through tunnel: `200/4xx/5xx` from your application (not `52x` Cloudflare origin errors), plus stable latency consistent with origin.
  - Origin unreachable: Cloudflare `52x` codes (`522`, `523`, `524`) or timeouts.
- Probe sequence (low-noise)
  1) DNS resolve target; record IPs.
  2) TLS handshake to target host; record cert fingerprint and ALPN.
  3) HEAD `/health` with `X-API-Key`; read headers only.
  4) If 401/403, retry HEAD `/` or a static asset path; else if 2xx/4xx, assume origin egress succeeded.
  5) Optional GET `/health` for body if needed.

### Multi-Signal Stream Design
- Signals (each optional, independently timed with jitter):
  - `signal_connectivity`: TCP/TLS handshake timing and cert data.
  - `signal_headers`: HTTP HEAD to `/health` (or `/`) to collect headers only.
  - `signal_health`: GET `/health` to parse minimal JSON.
  - `signal_tool_ping`: POST `/api/v1/tools/call` with `ping` payload when allowed.
  - `signal_ident_headers`: capture `Via`, `Server`, `cf-ray`, `x-request-id`, `traceparent` if present.
- Orchestration
  - Execute in sequence with backoff; cache successful results for N minutes to avoid repetition.
  - Emit a consolidated report with per-signal status and timestamps.

### Low-Profile/False-Positive Minimization
- Intent: reduce noise and false positives in WAF/bot systems, not to bypass access controls.
- Practices
  - Use HEAD instead of GET when content is unnecessary.
  - Reuse the same headers as the legitimate UI: `User-Agent`, `Accept`, `Content-Type`, `X-API-Key`.
  - Limit rate: at most 1 series per 1–5 minutes with random jitter; exponential backoff on errors.
  - Keep payload sizes small; avoid scanning multiple paths—stick to `/health` and one tool call.
  - Prefer HTTP/2 when available; avoid unusual client fingerprints.
  - Respect 4xx/5xx responses; do not retry aggressively on `403` or `429`.
  - Log correlation: include a harmless `traceparent`/`x-request-id` value to correlate on server logs when you control the origin.

### Updated Acceptance for Smart Ping
- Determine whether the probe reached Cloudflare edge.
- Determine whether the origin behind the tunnel was reached (egress succeeded) using status codes and header patterns.
- Produce a stable `signature` and a compact summary suitable for display in the UI.

