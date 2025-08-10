#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def probe_gpu() -> Dict[str, Any]:
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if p.returncode != 0:
            return {"available": False, "error": p.stderr.strip() or p.stdout.strip()}
        line = (p.stdout.strip().splitlines() or [""])[0]
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            name, mem_total, mem_used, util = parts[:4]
            total = int(mem_total)
            used = int(mem_used)
            utili = int(util)
            return {
                "available": True,
                "name": name,
                "memory_total_mb": total,
                "memory_used_mb": used,
                "utilization_pct": utili,
            }
        return {"available": True, "raw": line}
    except Exception as e:
        return {"available": False, "error": str(e)}


def probe_ollama() -> Dict[str, Any]:
    import requests  # type: ignore

    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    out: Dict[str, Any] = {"base_url": base}
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        out["tags_status_code"] = r.status_code
        out["models"] = (r.json() or {}).get("models", [])
        model = os.getenv("OLLAMA_MODEL")
        if model:
            # Try quick warmup with small num_predict and backoff
            timeouts = [int(os.getenv("OLLAMA_HEALTH_TIMEOUT", "20")), 40, 60]
            last_err: Optional[str] = None
            for to in timeouts:
                try:
                    t0 = time.time()
                    g = requests.post(
                        f"{base}/api/generate",
                        json={
                            "model": model,
                            "prompt": "ping",
                            "stream": False,
                            "options": {"num_predict": 8},
                        },
                        timeout=to,
                    )
                    out["generate_status_code"] = g.status_code
                    out["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
                    if g.ok:
                        out["ok"] = True
                        break
                    last_err = g.text[:200]
                except Exception as ge:
                    last_err = str(ge)
            if not out.get("ok"):
                out["ok"] = False
                if last_err:
                    out["error"] = last_err
        else:
            out["ok"] = True
        return out
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
        return out


def probe_pgvector() -> Dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from modules.module_2_support.postgresql_vector_agent import PostgreSQLVectorAgent  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"import error: {e}"}

    dsn = os.getenv("DATABASE_URL")
    dim = int(os.getenv("TEST_EMBED_DIM", "256"))
    if not dsn:
        return {"success": False, "error": "DATABASE_URL not set"}
    try:
        import asyncio

        async def run():
            agent = PostgreSQLVectorAgent(connection_string=dsn, embedding_dimension=dim)
            init = await agent.initialize()
            if not init.get("success"):
                return {"success": False, "error": init.get("error")}
            hc = await agent.health_check()
            await agent.close()
            return {"success": True, "health": hc}

        return asyncio.get_event_loop().run_until_complete(run())
    except Exception as e:
        return {"success": False, "error": str(e)}


def main() -> int:
    report = {
        "pgvector": probe_pgvector(),
        "ollama": probe_ollama(),
        "gpu": probe_gpu(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


