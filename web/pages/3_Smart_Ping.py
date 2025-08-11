from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.utils.smart_ping import probe_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    st.set_page_config(page_title="Smart Ping", layout="wide")
    st.title("Smart Ping")

    # Defaults from parent app if present
    api_base = (st.secrets.get("API_BASE") if hasattr(st, "secrets") else None) or os.getenv("API_BASE", "http://localhost:8000")
    api_key = (st.secrets.get("DEMO_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("DEMO_API_KEY", "demo_key_123")
    llm_key = st.session_state.get("LLM_KEY")

    url = st.text_input("Target URL", value=api_base)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        api_key = st.text_input("X-API-Key", value=api_key, type="password")
    with c2:
        llm_key = st.text_input("X-LLM-Key (optional)", value=llm_key or "", type="password")
    with c3:
        include_body = st.checkbox("Include /health body", value=False)

    if st.button("Run Smart Ping", type="primary"):
        with st.status("Probing target…", expanded=False):
            try:
                data = probe_target(url, api_key=api_key or None, llm_key=llm_key or None, include_health_body=include_body)
                st.success("Probe complete")
                # Summary cards
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Resolved IPs", ", ".join(data.get("resolved_ips") or []) or "(none)")
                with colB:
                    st.metric("TLS/ALPN", (data.get("tls") or {}).get("alpn") or "-")
                with colC:
                    st.metric("Signature", (data.get("signature") or "")[7:19] + "…")

                st.subheader("Raw Result")
                st.code(json.dumps(data, indent=2))
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption("Smart Ping collects DNS, TLS, and HTTP header signals with low-noise tactics. It does not bypass security controls.")


if __name__ == "__main__":
    main()


