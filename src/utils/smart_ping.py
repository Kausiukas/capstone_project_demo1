from __future__ import annotations

import hashlib
import json
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


@dataclass
class TLSInfo:
    version: Optional[str]
    alpn: Optional[str]
    cipher: Optional[str]
    cert_sha256: Optional[str]
    issuer: Optional[str]
    subject: Optional[str]
    not_before: Optional[str]
    not_after: Optional[str]


def _format_cert_time(dt) -> Optional[str]:
    try:
        # dt is datetime
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _tls_probe(host: str, port: int, timeout: float = 5.0) -> Tuple[float, TLSInfo]:
    start = time.perf_counter()
    # Resolve first IPv4/IPv6 available
    addr_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    # Prefer IPv4 if present, else use first
    ipv4 = [ai for ai in addr_info if ai[0] == socket.AF_INET]
    target = (ipv4[0] if ipv4 else addr_info[0])
    family, socktype, proto, _, sockaddr = target

    sock = socket.socket(family, socktype, proto)
    sock.settimeout(timeout)
    try:
        context = ssl.create_default_context()
        try:
            # Offer common ALPN protocols to learn server preference
            context.set_alpn_protocols(["h2", "http/1.1"])
        except Exception:
            pass
        wrapped = context.wrap_socket(sock, server_hostname=host)
        wrapped.connect(sockaddr)
        # After handshake
        elapsed = (time.perf_counter() - start) * 1000.0

        # Gather TLS details
        try:
            cert_bin = wrapped.getpeercert(binary_form=True)
        except Exception:
            cert_bin = None
        cert_sha256 = hashlib.sha256(cert_bin).hexdigest() if cert_bin else None
        try:
            peercert = wrapped.getpeercert()
        except Exception:
            peercert = {}
        try:
            cipher = wrapped.cipher()
            cipher_name = cipher[0] if cipher else None
        except Exception:
            cipher_name = None
        try:
            version = wrapped.version()
        except Exception:
            version = None
        try:
            alpn = wrapped.selected_alpn_protocol()
        except Exception:
            alpn = None

        issuer = None
        subject = None
        not_before = None
        not_after = None
        try:
            # peercert is dict with 'issuer', 'subject', 'notBefore', 'notAfter' in RFC 2822 format
            if isinstance(peercert, dict):
                # issuer/subject are tuples; join succinctly
                if peercert.get("issuer"):
                    issuer = ", ".join("=".join(x) for r in peercert["issuer"] for x in r)
                if peercert.get("subject"):
                    subject = ", ".join("=".join(x) for r in peercert["subject"] for x in r)
                not_before = peercert.get("notBefore")
                not_after = peercert.get("notAfter")
        except Exception:
            pass

        info = TLSInfo(
            version=version,
            alpn=alpn,
            cipher=cipher_name,
            cert_sha256=cert_sha256,
            issuer=issuer,
            subject=subject,
            not_before=not_before,
            not_after=not_after,
        )
        return elapsed, info
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _resolve_ips(host: str) -> List[str]:
    ips: List[str] = []
    try:
        for fam in (socket.AF_INET, socket.AF_INET6):
            try:
                for res in socket.getaddrinfo(host, None, family=fam, type=socket.SOCK_STREAM):
                    sockaddr = res[4]
                    ip = sockaddr[0]
                    if ip not in ips:
                        ips.append(ip)
            except Exception:
                continue
    except Exception:
        pass
    # Stabilize ordering for signature consistency
    try:
        ips.sort()
    except Exception:
        pass
    return ips


def _http_head(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> Tuple[int, Dict[str, str]]:
    try:
        r = requests.head(url, headers=headers or {}, timeout=timeout, allow_redirects=False)
        # Normalize header keys to canonical-lowercase
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        return r.status_code, hdrs
    except Exception:
        return 0, {}


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> Tuple[int, Any]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)


def _build_signature(host: str, ips: List[str], tls: Optional[TLSInfo], http_headers: Dict[str, str]) -> str:
    material = "|".join([
        host or "",
        (ips[0] if ips else ""),
        (tls.cert_sha256 if tls and tls.cert_sha256 else ""),
        (tls.alpn if tls and tls.alpn else ""),
        http_headers.get("server", ""),
    ])
    return "sha256:" + hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()


def probe_target(
    target_url: str,
    api_key: Optional[str] = None,
    llm_key: Optional[str] = None,
    include_health_body: bool = False,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Probe the target URL and collect connectivity, TLS, and HTTP header signals.

    Returns a JSON-serializable dictionary with a stable signature.
    """

    parsed = urlparse(target_url)
    if not parsed.scheme:
        raise ValueError("target_url must include scheme, e.g., https://host")
    host = parsed.hostname or ""

    # Resolve IPs
    ips = _resolve_ips(host) if host else []

    # TLS probe for HTTPS
    tls_elapsed_ms: Optional[float] = None
    tls_info: Optional[TLSInfo] = None
    if parsed.scheme == "https" and host:
        port = parsed.port or 443
        try:
            tls_elapsed_ms, tls_info = _tls_probe(host, port, timeout=timeout)
        except Exception:
            tls_info = None

    # Build headers that mirror the UI
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    if llm_key:
        headers["X-LLM-Key"] = llm_key
    # Use conservative UA to match requests default
    headers.setdefault("User-Agent", requests.utils.default_user_agent())

    # HEAD probe on /health if path unspecified; else HEAD on provided path
    base = f"{parsed.scheme}://{parsed.netloc}"
    head_url = base + "/health"
    head_status, head_headers = _http_head(head_url, headers=headers, timeout=timeout)

    # Cloudflare data
    cf_ray = head_headers.get("cf-ray")
    cf_colo = None
    if cf_ray and "-" in cf_ray:
        try:
            cf_colo = cf_ray.split("-")[-1]
        except Exception:
            cf_colo = None

    # Decide egress: if Cloudflare seen and non-52x from origin on GET → origin reached
    get_status: int = 0
    get_body: Any = None
    if include_health_body:
        get_status, get_body = _http_get_json(head_url, headers=headers, timeout=timeout)

    signature = _build_signature(host, ips, tls_info, head_headers)

    # Inferences
    server_hdr = (head_headers.get("server") or "").lower()
    edge_reached = ("cloudflare" in server_hdr) or (
        (getattr(tls_info, "subject", "") or "").find("trycloudflare.com") >= 0
    )
    origin_reached = False
    origin_reason = ""
    if head_status == 405:
        origin_reached = True
        origin_reason = "HEAD not allowed on /health (likely origin behavior)"
    if include_health_body and isinstance(get_body, (dict, list, str)) and get_status:
        # 2xx/4xx from origin with JSON/text body implies origin processed the request
        if 200 <= get_status < 600 and get_status not in (520, 521, 522, 523, 524):
            origin_reached = True
            if not origin_reason:
                origin_reason = f"GET /health returned {get_status}"

    result: Dict[str, Any] = {
        "target_url": target_url,
        "resolved_ips": ips,
        "tcp_tls_ms": tls_elapsed_ms,
        "tls": {
            "version": getattr(tls_info, "version", None),
            "alpn": getattr(tls_info, "alpn", None),
            "cipher": getattr(tls_info, "cipher", None),
            "cert_sha256": getattr(tls_info, "cert_sha256", None),
            "issuer": getattr(tls_info, "issuer", None),
            "subject": getattr(tls_info, "subject", None),
            "not_before": getattr(tls_info, "not_before", None),
            "not_after": getattr(tls_info, "not_after", None),
        },
        "http_head": {
            "url": head_url,
            "status": head_status,
            "headers": head_headers,
        },
        "http_health": {
            "status": get_status,
            "body": get_body if include_health_body else None,
        },
        "cloudflare": {
            "cf_ray": cf_ray,
            "colo": cf_colo,
        },
        "inferences": {
            "edge_reached": edge_reached,
            "origin_reached": origin_reached,
            "reason": origin_reason or None,
        },
        "signature": signature,
    }
    return result


def main_cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    import os
    from datetime import datetime
    parser = argparse.ArgumentParser(description="Smart Ping probe")
    parser.add_argument("--url", required=True, help="Target base URL, e.g., https://host")
    parser.add_argument("--api-key", default=os.getenv("DEMO_API_KEY", "demo_key_123"))
    parser.add_argument("--llm-key", default=os.getenv("LLM_KEY"))
    parser.add_argument("--include-body", action="store_true", help="Include /health body in output")
    parser.add_argument("--out-dir", default="results/ping_trace", help="Directory to save JSON output")
    args = parser.parse_args(argv)

    data = probe_target(
        target_url=args.url,
        api_key=args.api_key,
        llm_key=args.llm_key,
        include_health_body=args.include_body,
    )
    # Print to stdout
    print(json.dumps(data, indent=2))

    # Save to file
    try:
        import pathlib
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        host = urlparse(args.url).hostname or "host"
        out_path = out_dir / f"smart_ping_{host}_{ts}.json"
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())


