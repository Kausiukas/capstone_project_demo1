---
title: Website optimisation and instrumentation guide
layer: 3
purpose: Make the e‑shop a responsive resource for the agent; enable telemetry → KPIs → goals → planning
version: 1.0
updated: 2025-08-09
target_product_url: https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing
---

## Overview

This document gives step‑by‑step instructions to prepare the website so the agent can observe conversions, learn, and optimise towards revenue goals. Target page: [Bioplastics for sustainable manufacturing](https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing).

## Prerequisites

- Decide which checkout signal you can emit:
  - Preferred: platform server‑side webhooks for order/checkout.
  - Fallback: client‑side tracker on the “thank‑you” page.
- Pick your agent endpoint base URL (e.g., `https://agent.example.com`).
- Generate a shared secret for signing events and set it in the agent: `REVENUE_WEBHOOK_SECRET`.
- Ensure the agent exposes `POST /events/checkout`.

## 1) Implement reliable checkout event delivery

### A. Server‑side webhook (preferred)

1. In your e‑commerce platform, add a webhook for order/checkout completed events.
2. Point it to: `POST https://<agent-host>/events/checkout`.
3. Configure an HMAC signature using `REVENUE_WEBHOOK_SECRET`.
4. Send the following JSON body (fields marked R are required):
   - R `order_id` (string)
   - R `product_url` = `https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing`
   - R `product_id` (SKU)
   - R `quantity` (int)
   - R `currency` (e.g., `EUR`)
   - R `value` (numeric order total)
   - `customer_country`, `ts` (ISO 8601), `ua`, `src`
5. Include headers:
   - `X-Signature: sha256=<hex(hmac_sha256(secret, body))>`
   - `X-Timestamp: <epoch_ms>`
   - Optional `X-Nonce: <random-uuid>` (prevents replay)

Verification on the agent side rejects invalid signatures, stale timestamps (>15 min), or reused nonces.

### B. Client‑side tracker (fallback)

Add this snippet on the “thank‑you” page (fill real values):

```html
<script>
(async () => {
  const payload = {
    order_id: window.orderId || crypto.randomUUID(),
    product_url: "https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing",
    product_id: window.productId || "bioplastics-001",
    quantity: window.qty || 1,
    currency: "EUR",
    value: window.orderTotal || 0,
    customer_country: window.country || undefined,
    ts: new Date().toISOString(),
    ua: navigator.userAgent
  };
  await fetch("https://<agent-host>/events/checkout", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload),
    keepalive: true
  });
})();
</script>
```

Note: client tracking is less reliable and easier to spoof; migrate to server webhooks when possible.

## 2) Product page metadata for attribution and SEO

Add canonical, Open Graph, and JSON‑LD Product data. Example JSON‑LD:

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Bioplastics for sustainable manufacturing",
  "url": "https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing",
  "sku": "bioplastics-001",
  "image": ["https://example.com/path/to/image.jpg"],
  "description": "Eco-friendly bioplastics for manufacturing applications.",
  "brand": {"@type":"Brand","name":"EkoPolimeras"},
  "offers": {
    "@type": "Offer",
    "priceCurrency": "EUR",
    "price": "99.00",
    "availability": "https://schema.org/InStock",
    "url": "https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing"
  }
}
</script>
```

## 3) Conversion UX essentials

- Above‑the‑fold value statement, clear CTA (“Buy now”), primary button color.
- Trust signals: payment badges, secure checkout note, returns policy, contact info.
- Price clarity (incl. tax/shipping notes), shipping ETA on product page.
- Sticky add‑to‑cart on mobile; minimal checkout steps; guest checkout enabled.
- Accessibility: semantic buttons, focus styles, ARIA labels; text contrast ≥ 4.5:1.

## 4) Performance (Core Web Vitals)

- Target: LCP < 2.5s, CLS < 0.1, INP < 200ms.
- Optimise images (responsive sizes, modern formats, lazy‑load below fold).
- Preconnect to critical origins (CDN, payment); HTTP/2 or HTTP/3 enabled.
- Cache static assets with long max‑age; enable compression (brotli/gzip).
- Run Lighthouse; fix regressions before deploying experiments.

## 5) Analytics mapping to agent KPIs

Events the agent expects (names suggested; use your analytics tooling as desired):

- `view_product` (product_url)
- `add_to_cart` (product_id, qty)
- `begin_checkout`
- `purchase` (order_id, product_id, value, currency)

These map to KPIs in goals.yaml: `checkouts_per_day`, `conversion_rate`, `aov_eur`, `revenue_per_week`, `p95_checkout_ms`.
If you use a tag manager, push a `dataLayer` event mirroring the payload posted to `/events/checkout`.

## 6) Security & privacy

- Use HMAC signatures and timestamp/nonce on webhooks; rotate `REVENUE_WEBHOOK_SECRET` regularly.
- Rate‑limit `/events/checkout`; optionally IP allowlist platform webhook origin.
- Send only non‑sensitive data (no raw PII). Store order_id and aggregates, not personal details.
- Maintain Privacy Policy and Cookie consent banner if analytics writes cookies.

## 7) Monitoring & alerting

- Uptime checks: product page URL and `/events/checkout` endpoint (200 responses, body JSON).
- Alert if checkouts drop to 0 during configured business hours.
- Log anomalies: invalid signatures, replay attempts, error spikes.

## 8) A/B testing approach (safe)

- Budget guard: weekly cap (see goals.yaml `constraints.max`).
- Test one variable at a time (headline, hero image, CTA copy, pricing display).
- Stop early if conversion lifts ≥ pre‑set threshold or error/latency regresses.

## 9) Verification checklist

- [ ] POST a signed sample to `/events/checkout` → 200 OK
- [ ] Event appears in agent logs and database table `checkout_events`
- [ ] `tasklog_dashboard.py --auto --write` shows updated revenue KPIs
- [ ] L3 prioritisation favours revenue when below targets
- [ ] Lighthouse shows LCP/CLS/INP within targets on product page
- [ ] Accessibility quick audit passes (keyboard, labels, contrast)

## 10) Maintenance

- Version your tracker snippet; keep change log.
- Re‑run performance and accessibility audits after content updates.
- Review KPIs weekly; tune `goals.yaml` weights/targets based on progress.

## References

- Product page: [Bioplastics for sustainable manufacturing](https://ekopolimeras.shop/bioplastics-for-sustainable-manufacturing)


