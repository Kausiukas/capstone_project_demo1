# API Identificator Design

## Problem
When accessing the API through a tunnel or proxy (e.g., Cloudflare), network-level fingerprints identify the edge, not the backend. We need a reliable, low-noise way to determine which backend API instance we reached, and to expose a stable unique identifier across builds and deployments.

## Goals
- Provide a standardized identity object via a dedicated endpoint and in key responses (health, ping).
- Expose identity headers on all responses for easy retrieval without parsing bodies.
- Maintain uniqueness across builds and restarts.
- Keep traffic low-profile and consistent with normal application use.

## Solution Overview
- Identity object builder on the server: combines env vars, optional identity.json overrides, and computed fields.
- New endpoint: `GET /identity` returning the full identity object.
- Augmented existing endpoints:
  - `GET /health` now includes `instance: { … }` with identity.
  - `POST /api/v1/tools/call` ping returns `instance`.
- Response headers added to every request: `X-Api-Hostname`, `X-Api-Started-At`, `X-Api-Image-Tag`, `X-Api-Commit`, `X-Api-Instance-Id`.
- Smart Ping updated to call `/identity` alongside `/health` for explicit identity.

## Identity Object Fields
- `service`: logical service name (default `capstone-demo-api`, override via SERVICE_NAME or identity.json).
- `version`: service version (default `0.1.0`, override via SERVICE_VERSION or identity.json).
- `hostname`: container or host name.
- `started_at`: process start epoch seconds.
- `image_tag`: container/image tag.
- `commit_sha`: source commit (short or full).
- `build_time`: ISO/RFC value if available.
- `instance_id`: short hash derived from `(hostname|started_at|image_tag|commit_sha|build_time)`.
- `identificator`: human-provided identifier; if not provided, equals `instance_id`.
- `python`, `pid`: runtime info for debugging.
- `extras`: arbitrary key/values from identity.json.

## Uniqueness Guarantees
- `instance_id` changes on any of: hostname change, rebuild (image_tag/commit/build_time), or restart (started_at).
- Distinct builds share no `instance_id`; restarts change it as well, which is useful for freshness detection.

## Cloudflare Considerations
- Use Transform Rules or Workers to ensure identity headers from origin are preserved in responses.
- Expression example: `http.host eq "<your-tunnel>.trycloudflare.com"` and set headers from `http.response.headers`.

## Local Discovery of Ping Builds
- Goal: find all local ping implementations and include identity in their responses.
- Strategy:
  1) Grep project for `"ping"` tool endpoints or handlers (e.g., `/api/v1/tools/call` with name == "ping").
  2) For each handler, ensure it returns `instance: build_identity()`.
  3) For any legacy endpoints, add a thin wrapper to call `build_identity()`.
- In this repo: `api/main.py` ping already returns `instance` via `build_identity()`.

## Standardized Script
- CLI `scripts/smart_ping.py` collects:
  - DNS/TLS/HTTP signals
  - `HEAD /health` headers
  - `GET /health` body (optional)
  - `GET /identity` body
- Output JSON contains a stable `signature` and explicit identity if exposed by the server.

## Usage Examples
- CLI:
  - `python scripts/smart_ping.py --url https://<tunnel>.trycloudflare.com --include-body`
  - `python scripts/smart_ping.py --url http://localhost:8000 --include-body`
- UI: Tools/APIs → Smart Ping section or dedicated Smart Ping page.

## Definition of Done
- `/identity` returns identity object; `/health` and ping include `instance` with the same object.
- Identity headers are present on all responses.
- Smart Ping displays identity headers and the `/identity` body when available.


