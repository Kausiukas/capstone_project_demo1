import os
import json
import pytest

import requests


API = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", os.getenv("DEMO_API_KEY", "demo_key_123"))


def _headers():
    return {"X-API-Key": API_KEY}


@pytest.mark.timeout(10)
def test_identity_endpoint():
    r = requests.get(f"{API}/identity", headers=_headers(), timeout=5)
    assert r.status_code == 200
    data = r.json()
    for k in ("service", "hostname", "instance_id"):
        assert k in data and data[k]


@pytest.mark.timeout(10)
def test_health_includes_instance():
    r = requests.get(f"{API}/health", headers=_headers(), timeout=5)
    assert r.status_code == 200
    d = r.json()
    inst = d.get("instance", {})
    assert isinstance(inst, dict) and inst.get("instance_id")


@pytest.mark.timeout(15)
def test_ping_tool_returns_components():
    r = requests.post(
        f"{API}/api/v1/tools/call",
        json={"name": "ping", "arguments": {}},
        headers=_headers(),
        timeout=10,
    )
    assert r.status_code == 200
    d = r.json()
    assert d.get("content") and d["content"][0]["text"] == "pong"
    assert d.get("instance") and d["instance"].get("instance_id")
    assert isinstance(d.get("llm"), dict)
    assert isinstance(d.get("memory"), dict)
    assert isinstance(d.get("endpoints"), dict)


