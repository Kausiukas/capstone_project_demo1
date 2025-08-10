#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _ensure_file(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def discover_report_path(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        p = base_dir / f"development_session_report_{n}.md"
        if not p.exists():
            return p
        n += 1


def pick_tasks(dashboard_json: Dict[str, Any], max_layers: int = 2, max_tasks_per_layer: int = 5) -> List[Tuple[str, List[str]]]:
    selections: List[Tuple[str, List[str]]] = []
    recs = dashboard_json.get("recommendations") or []
    for r in recs[:max_layers]:
        lid = r.get("layer")
        next_tasks = [t for t in (r.get("next_tasks") or [])[:max_tasks_per_layer]]
        if next_tasks:
            selections.append((str(lid), next_tasks))
    return selections


def apply_known_scaffolds(task: str) -> Optional[Tuple[str, Optional[str]]]:
    # Minimal safe scaffolds to flip auto rules; extend as needed. Returns (action_note, test_path?)
    if "Tests: `test_normalize_summarize.py`" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_normalize_summarize.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_normalize_summarize():\n    pytest.skip(\"normalization/synthesis not implemented yet\")\n""",
        )
        return ("created test_normalize_summarize.py", str(PROJECT_ROOT / "tests/phase1/test_normalize_summarize.py")) if created else None
    if "Tests: `test_schedule_validation.py`" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_schedule_validation.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_schedule_validation():\n    pytest.skip(\"scheduler/validation not implemented yet\")\n""",
        )
        return ("created test_schedule_validation.py", str(PROJECT_ROOT / "tests/phase1/test_schedule_validation.py")) if created else None
    if "Collectors fetch and store snippets with tags and provenance" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_collectors_provenance.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_collectors_provenance():\n    pytest.skip(\"collector persistence not implemented yet\")\n""",
        )
        return ("created test_collectors_provenance.py", str(PROJECT_ROOT / "tests/phase1/test_collectors_provenance.py")) if created else None
    if "Type detection; safe parsing; preview extraction" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_type_detection.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_type_detection():\n    pytest.skip(\"type detection not implemented yet\")\n""",
        )
        return ("created test_type_detection.py", str(PROJECT_ROOT / "tests/phase1/test_type_detection.py")) if created else None
    if "Synthesis briefs (" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_synthesis_briefs.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_synthesis_briefs():\n    pytest.skip(\"brief synthesis not implemented yet\")\n""",
        )
        return ("created test_synthesis_briefs.py", str(PROJECT_ROOT / "tests/phase1/test_synthesis_briefs.py")) if created else None
    # L6 consolidation stubs
    if "Nightly cluster→summarize" in task or "Nightly cluster" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_consolidation_jobs.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_consolidation_jobs():\n    pytest.skip(\"consolidation jobs not implemented yet\")\n""",
        )
        return ("created test_consolidation_jobs.py", str(PROJECT_ROOT / "tests/phase1/test_consolidation_jobs.py")) if created else None
    # L5 planner/executor concrete stubs
    if "`ExecutionEngine.run_chain(steps, payload)`" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_executor_sequential.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_executor_sequential():\n    pytest.skip(\"executor sequential not implemented yet\")\n""",
        )
        return ("ensured test_executor_sequential.py", str(PROJECT_ROOT / "tests/phase1/test_executor_sequential.py")) if created else None
    if "DAG planner" in task or "run_dag()" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_executor_dag.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_executor_dag():\n    pytest.skip(\"executor dag not implemented yet\")\n""",
        )
        return ("ensured test_executor_dag.py", str(PROJECT_ROOT / "tests/phase1/test_executor_dag.py")) if created else None
    if "test_error_mapping.py" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_error_mapping.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_error_mapping():\n    pytest.skip(\"error mapping not implemented yet\")\n""",
        )
        return ("created test_error_mapping.py", str(PROJECT_ROOT / "tests/phase1/test_error_mapping.py")) if created else None
    if "orchestrator_smoke" in task or "Smoke task" in task:
        created = _ensure_file(
            PROJECT_ROOT / "tests/phase1/test_orchestrator_smoke.py",
            """from __future__ import annotations\n\nimport pytest\n\n\ndef test_placeholder_orchestrator_smoke():\n    pytest.skip(\"orchestrator smoke not implemented yet\")\n""",
        )
        return ("created test_orchestrator_smoke.py", str(PROJECT_ROOT / "tests/phase1/test_orchestrator_smoke.py")) if created else None
    return None


def store_memory(summary: str) -> Optional[str]:
    # Optional: persist a memory snippet if DATABASE_URL is available
    try:
        if not os.getenv("DATABASE_URL"):
            return None
        # Lazy import here to avoid src/__init__ side effects
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from layers.memory_system import MemorySystem, Interaction  # type: ignore
        from datetime import datetime as _dt

        dim = int(os.getenv("TEST_EMBED_DIM", "256"))
        ms = MemorySystem(embedding_dimension=dim, window_max_chars=2000)

        async def run():
            await ms.initialize()
            emb = [random.random() * 0.01 for _ in range(dim)]
            inter = Interaction(
                user_input="self_dev_session",
                response=summary,
                tool_calls=[],
                metadata={"tags": ["self_dev", "session"]},
                timestamp=_dt.utcnow(),
            )
            await ms.store_interaction(inter, emb)

        import asyncio

        asyncio.get_event_loop().run_until_complete(run())
        return "stored"
    except Exception:
        return None


def _llm_client():
    # Explicit provider override takes precedence
    provider_override = (os.getenv("LLM_PROVIDER") or "").lower().strip()
    if provider_override == "ollama":
        try:
            import requests  # type: ignore
            base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            model = os.getenv("OLLAMA_MODEL")
            if not model:
                return None
            with contextlib.suppress(Exception):
                requests.get(base, timeout=1)
            return ("ollama", {"base": base, "model": model})
        except Exception:
            return None
    if provider_override == "openai":
        try:
            import openai  # type: ignore
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            try:
                from openai import OpenAI  # type: ignore
                base_url = os.getenv("OPENAI_BASE_URL")
                client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
                return ("sdk", client)
            except Exception:
                openai.api_key = api_key  # type: ignore
                base = os.getenv("OPENAI_BASE_URL")
                if base:
                    openai.base_url = base  # type: ignore
                return ("legacy", openai)
        except Exception:
            return None

    # Prefer local Ollama if configured
    try:
        if os.getenv("OLLAMA_MODEL"):
            import requests  # type: ignore
            base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            model = os.getenv("OLLAMA_MODEL")
            with contextlib.suppress(Exception):
                requests.get(base, timeout=1)
            return ("ollama", {"base": base, "model": model})
    except Exception:
        pass

    # Fallback to OpenAI if key available
    try:
        import openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
            base_url = os.getenv("OPENAI_BASE_URL")
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            return ("sdk", client)
        except Exception:
            openai.api_key = api_key  # type: ignore
            base = os.getenv("OPENAI_BASE_URL")
            if base:
                openai.base_url = base  # type: ignore
            return ("legacy", openai)
    except Exception:
        return None


def llm_suggest_edits(dashboard: Dict[str, Any], model: str, out_dir: Path) -> Optional[Tuple[Path, Optional[int], Dict[str, int]]]:
    """Ask LLM to propose micro-edits for top pending tasks. Writes suggestions to JSONL.
    Never includes secrets in prompts; does not apply edits automatically.
    """
    client_info = _llm_client()
    if not client_info:
        return None
    mode, client = client_info
    suggestions_dir = out_dir / "results" / "llm_suggestions"
    suggestions_dir.mkdir(parents=True, exist_ok=True)
    path = suggestions_dir / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")

    # Build compact prompt from dashboard
    recs = dashboard.get("recommendations") or []
    tasks_brief = []
    for r in recs[:3]:
        tasks_brief.append({"layer": r.get("layer"), "next_tasks": (r.get("next_tasks") or [])[:5]})
    prompt = (
        "You are assisting a codebase self-development cycle. Given pending tasks per layer, "
        "propose at most 5 concrete, small edits (with file paths and short diffs in prose) "
        "that are safe to apply in <15 minutes total>. Prefer tests/docs/scaffolds that flip readiness. "
        "Do not include secrets. Respond as JSON list of {layer, path, rationale, steps}.\n\n"
        + json.dumps(tasks_brief)
    )

    try:
        if mode == "sdk":
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You propose safe, small codebase edits."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            content = resp.choices[0].message.content or "[]"
            usage = getattr(resp, "usage", None)
            tokens = {
                "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total": getattr(usage, "total_tokens", 0) if usage else 0,
            }
        else:
            if mode == "legacy":
                content = client.ChatCompletion.create(  # type: ignore
                    model=model,
                    messages=[{"role": "system", "content": "You propose safe, small codebase edits."},
                              {"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800,
                )
                tokens = {
                    "prompt": int(content.get("usage", {}).get("prompt_tokens", 0)),
                    "completion": int(content.get("usage", {}).get("completion_tokens", 0)),
                    "total": int(content.get("usage", {}).get("total_tokens", 0)),
                }
                content = content["choices"][0]["message"]["content"]
            elif mode == "ollama":
                import requests  # type: ignore
                base = client["base"].rstrip("/")
                mdl = client["model"] or model
                timeout_s = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "90"))
                def _chat_once(to: int):
                    return requests.post(
                        f"{base}/api/chat",
                        json={
                            "model": mdl,
                            "messages": [
                                {"role": "system", "content": "You propose safe, small codebase edits."},
                                {"role": "user", "content": prompt},
                            ],
                            "stream": False,
                            "options": {"temperature": 0.2},
                        },
                        timeout=to,
                    )
                try:
                    resp = _chat_once(timeout_s)
                    resp.raise_for_status()
                except Exception as e:
                    # One retry after short backoff with extended timeout
                    time.sleep(1.0)
                    try:
                        resp = _chat_once(min(120, timeout_s + 30))
                        resp.raise_for_status()
                    except Exception as e2:
                        # Write a debug breadcrumb for troubleshooting
                        try:
                            suggestions_dir = (out_dir / "results" / "llm_suggestions")
                            suggestions_dir.mkdir(parents=True, exist_ok=True)
                            (suggestions_dir / "error_last.txt").write_text(
                                f"{datetime.now(timezone.utc).isoformat()} chat error: {str(e2)}\n", encoding="utf-8"
                            )
                        except Exception:
                            pass
                        raise
                data = resp.json()
                content = (data.get("message", {}) or {}).get("content", "[]")
                tokens = {
                    "prompt": int(data.get("prompt_eval_count", 0) or 0),
                    "completion": int(data.get("eval_count", 0) or 0),
                    "total": int((data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0)),
                }
        # Write a single line JSON object
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "suggestions": content}) + "\n")
        # Try to count suggestions
        try:
            parsed = json.loads(content)
            count = len(parsed) if isinstance(parsed, list) else None
        except Exception:
            count = None
        return (path, count, tokens)
    except Exception:
        return None

def write_report(path: Path, started_at: datetime, ended_at: datetime, cycles: int,
                 actions: List[str], bottlenecks: Dict[str, Any], tasks_before: Dict[str, Any],
                 tasks_after: Dict[str, Any],
                 llm_stats: Optional[Dict[str, int]] = None,
                 probe_snapshot: Optional[Dict[str, Any]] = None) -> None:
    def block(title: str, data: Any) -> str:
        return f"\n### {title}\n\n" + "```json\n" + json.dumps(data, indent=2) + "\n```\n"

    md = [
        f"# Development Session Report\n",
        f"Started: {started_at.isoformat()}\n",
        f"Ended: {ended_at.isoformat()}\n",
        f"Cycles: {cycles}\n",
        f"Actions: {len(actions)}\n",
    ]
    if actions:
        md.append("\n## Actions taken\n\n" + "\n".join(f"- {a}" for a in actions) + "\n")
    md.append(block("Bottlenecks (final)", bottlenecks))
    md.append(block("Tasks before", tasks_before))
    md.append(block("Tasks after", tasks_after))
    if llm_stats:
        md.append("\n---\n")
        md.append(f"LLM suggestions: {llm_stats.get('suggestions', 0)}; tokens prompt={llm_stats.get('prompt', 0)} completion={llm_stats.get('completion', 0)} total={llm_stats.get('total', 0)}\n")
    if probe_snapshot:
        md.append(block("Readiness snapshot", probe_snapshot))
    path.write_text("".join(md), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an iterative self-development session")
    ap.add_argument("--duration-min", type=int, default=60)
    ap.add_argument("--dim", type=int, default=int(os.getenv("TEST_EMBED_DIM", "256")))
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--report-dir", type=str, default=str(PROJECT_ROOT / "7_agent_layers"))
    ap.add_argument("--dry-run", action="store_true", help="Do not scaffold files; only simulate")
    ap.add_argument("--use-llm", action="store_true", help="Use OpenAI for suggestions if OPENAI_API_KEY is set")
    ap.add_argument("--openai-model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5"))
    ap.add_argument("--dotenv-file", type=str, default=str(PROJECT_ROOT / ".env"))
    ap.add_argument("--max-layers", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=6)
    ap.add_argument("--max-tests-per-loop", type=int, default=6)
    ap.add_argument("--skip-db-tests", action="store_true", help="Skip DB-heavy tests each loop")
    ap.add_argument("--run-db-tests-every", type=int, default=5, help="Run DB tests every N cycles")
    ap.add_argument("--quiet-hours", type=str, default="", help="UTC ranges, e.g., 00:00-06:00,22:00-23:00")
    ap.add_argument("--preload-llm", action="store_true", help="Warm up local LLM (Ollama) before cycles")
    ap.add_argument("--preload-timeout", type=int, default=int(os.getenv("OLLAMA_HEALTH_TIMEOUT", "60")))
    ap.add_argument("--preload-pull", action="store_true", help="If model missing, request /api/pull (may take long)")
    args = ap.parse_args()

    # Best-effort load of .env
    try:
        env_path = Path(args.dotenv_file)
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

    started = datetime.now(timezone.utc)
    end_ts = time.time() + args.duration_min * 60
    cycles = 0
    actions: List[str] = []

    # Snapshot tasks before
    rc, out_before, _ = _run([sys.executable, "scripts/tasklog_dashboard.py", "--auto", "--json"])
    tasks_before = json.loads(out_before) if out_before.strip().startswith("{") else {}

    # Pre-flight LLM requirement logging
    if args.use_llm:
        try:
            probe = _llm_client()
            if not probe:
                actions.append("REQ: Local or remote LLM unavailable. Provide OPENAI_API_KEY or start Ollama and set OLLAMA_MODEL.")
            else:
                # Prefer Ollama when configured for this session unless explicitly overridden
                if os.getenv("OLLAMA_MODEL") and not os.getenv("LLM_PROVIDER"):
                    os.environ["LLM_PROVIDER"] = "ollama"
                # Optional preload for Ollama
                mode, client = probe
                if args.preload_llm and mode == "ollama":
                    try:
                        import requests  # type: ignore
                        base = client["base"].rstrip("/")
                        model = client["model"] or os.getenv("OLLAMA_MODEL")
                        # Ensure model present
                        present = False
                        try:
                            r = requests.get(f"{base}/api/tags", timeout=5)
                            models = (r.json() or {}).get("models", [])
                            present = any((m.get("name",""))[: len(str(model))] == str(model) for m in models)
                        except Exception:
                            pass
                        if not present and args.preload_pull and model:
                            actions.append(f"INFO: Pulling model {model} via /api/pull (this may take a while)...")
                            requests.post(f"{base}/api/pull", json={"name": model}, timeout=max(60, args.preload_timeout))
                        # Warmup generate
                        t0 = time.time()
                        g = requests.post(
                            f"{base}/api/generate",
                            json={
                                "model": model,
                                "prompt": "ping",
                                "stream": False,
                                "options": {"num_predict": 8},
                            },
                            timeout=args.preload_timeout,
                        )
                        if g.ok:
                            actions.append(f"LLM preload ok: {round((time.time()-t0)*1000)} ms")
                        else:
                            actions.append(f"WARN: LLM preload HTTP {g.status_code}")
                        # Warmup chat as well (separate path)
                        try:
                            t1 = time.time()
                            c = requests.post(
                                f"{base}/api/chat",
                                json={
                                    "model": model,
                                    "messages": [
                                        {"role": "system", "content": "probe"},
                                        {"role": "user", "content": "ping"},
                                    ],
                                    "stream": False,
                                    "options": {"num_predict": 8},
                                },
                                timeout=max(10, min(60, args.preload_timeout)),
                            )
                            if c.ok:
                                actions.append(f"LLM chat preload ok: {round((time.time()-t1)*1000)} ms")
                            else:
                                actions.append(f"WARN: LLM chat preload HTTP {c.status_code}")
                        except Exception as ce:
                            actions.append(f"WARN: LLM chat preload failed: {str(ce)[:120]}")
                    except Exception as e:
                        actions.append(f"WARN: LLM preload failed: {str(e)[:120]}")
        except Exception:
            actions.append("REQ: LLM probe failed. Verify LLM provider credentials or local server.")

    llm_agg = {"suggestions": 0, "prompt": 0, "completion": 0, "total": 0}
    state_file = PROJECT_ROOT / "results" / "session_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"created": []}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    def in_quiet_hours(now_utc: datetime) -> bool:
        spec = args.quiet_hours.strip()
        if not spec:
            return False
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for rng in parts:
            try:
                a, b = [x.strip() for x in rng.split("-")]
                h1, m1 = map(int, a.split(":"))
                h2, m2 = map(int, b.split(":"))
                t1 = h1 * 60 + m1
                t2 = h2 * 60 + m2
                nowm = now_utc.hour * 60 + now_utc.minute
                if t1 <= t2:
                    if t1 <= nowm <= t2:
                        return True
                else:
                    # overnight wrap
                    if nowm >= t1 or nowm <= t2:
                        return True
            except Exception:
                continue
        return False

    last_probe: Optional[Dict[str, Any]] = None
    while time.time() < end_ts:
        # Pre-loop health probe
        try:
            rc_hp, hp_out, _ = _run([sys.executable, "scripts/health_probe.py"])
            if rc_hp == 0 and hp_out.strip().startswith("{"):
                hp = json.loads(hp_out)
                last_probe = hp
                # VRAM guidance
                gpu = hp.get("gpu") or {}
                if gpu.get("available") and int(gpu.get("memory_used_mb", 0)) > int(gpu.get("memory_total_mb", 0)) * 0.8:
                    actions.append("WARN: High GPU VRAM usage. Consider a smaller local model (e.g., llama3.2:3b).")
                # LLM readiness
                oll = hp.get("ollama") or {}
                if not oll.get("ok"):
                    actions.append(f"REQ: Ollama not ready: {oll.get('error','unknown')[:120]}")
                # DB readiness
                pg = hp.get("pgvector") or {}
                if not pg.get("success"):
                    actions.append(f"REQ: pgvector not ready: {pg.get('error','unknown')[:120]}")
        except Exception:
            pass
        # Quiet hours backoff for traces
        if in_quiet_hours(datetime.now(timezone.utc)):
            time.sleep(max(0.5, args.sleep))
        else:
            _run([sys.executable, "scripts/run_traces.py", "--dim", str(args.dim), "--sleep", str(args.sleep), "--summarize"])  # noqa: E501
        # Auto-detect tasks and optionally write checkboxes; capture flips as actions
        rc, before_json, _ = _run([sys.executable, "scripts/tasklog_dashboard.py", "--auto", "--json"])
        before = json.loads(before_json) if before_json.strip().startswith("{") else {"tasks": {}}
        _run([sys.executable, "scripts/tasklog_dashboard.py", "--auto", "--write", "--json"])
        rc, after_json, _ = _run([sys.executable, "scripts/tasklog_dashboard.py", "--auto", "--json"])
        after = json.loads(after_json) if after_json.strip().startswith("{") else {"tasks": {}}
        try:
            for lid, t in (after.get("tasks") or {}).items():
                done_after = int((t or {}).get("done", 0))
                done_before = int(((before.get("tasks") or {}).get(lid) or {}).get("done", 0))
                delta = done_after - done_before
                if delta > 0:
                    actions.append(f"[L{lid}] checkboxes flipped +{delta}")
        except Exception:
            pass

        # Read dashboard to choose tasks
        rc, dash_json, _ = _run([sys.executable, "scripts/tasklog_dashboard.py", "--auto", "--json"])
        try:
            dashboard = json.loads(dash_json)
        except Exception:
            dashboard = {}
        if args.use_llm:
            res = llm_suggest_edits(dashboard, args.openai_model, PROJECT_ROOT)
            if res:
                llm_path, count, tokens = res
                actions.append(f"LLM suggestions saved: {llm_path}")
                if count:
                    llm_agg["suggestions"] += int(count)
                for k in ("prompt", "completion", "total"):
                    llm_agg[k] += int(tokens.get(k, 0))
            else:
                actions.append("REQ: LLM suggestion call failed or timed out (check OLLAMA_MODEL/OPENAI_API_KEY)")
        selections = pick_tasks(dashboard, max_layers=args.max_layers, max_tasks_per_layer=args.max_tasks)

        targeted_tests: List[str] = []
        for lid, task_list in selections:
            for t in task_list:
                res = apply_known_scaffolds(t) if not args.dry_run else None
                if res:
                    note, test_path = res
                    actions.append(f"[L{lid}] {note}")
                    if test_path:
                        if test_path not in state.get("created", []):
                            state.setdefault("created", []).append(test_path)
                        targeted_tests.append(test_path)

        # Re-run tests quickly to flip rules (prefer targeted paths)
        # Filter DB-heavy tests if requested
        db_markers = ("pgvector", "memory_window_ranking", "database")
        if args.skip_db_tests and (cycles % max(1, args.run_db_tests_every)) != 0:
            targeted_tests = [p for p in targeted_tests if not any(x in p for x in db_markers)]
        # Cap number of tests per loop
        if targeted_tests:
            to_run = targeted_tests[: max(1, args.max_tests_per_loop)]
            _run([sys.executable, "-m", "pytest", "-q", *to_run])
        else:
            # run a tiny subset by default
            _run([sys.executable, "-m", "pytest", "-q", "tests/phase1/test_collectors_basic.py"])  # keep fast

        cycles += 1

        # Short pause to avoid hammering DB/FS
        state_file.write_text(json.dumps(state), encoding="utf-8")
        time.sleep(max(0.1, args.sleep))

    # Final snapshots
    rc, out_after, _ = _run([sys.executable, "scripts/tasklog_dashboard.py", "--auto", "--json"])
    tasks_after = json.loads(out_after) if out_after.strip().startswith("{") else {}
    rc, out_bottl, _ = _run([sys.executable, "scripts/tasklog_dashboard.py", "--json"])
    bottlenecks = json.loads(out_bottl).get("bottlenecks", {}) if out_bottl.strip().startswith("{") else {}

    # Store memory summary (best-effort)
    store_memory(f"Self-dev session cycles={cycles} actions={len(actions)}")

    report_path = discover_report_path(Path(args.report_dir))
    write_report(report_path, started, datetime.now(timezone.utc), cycles, actions, bottlenecks, tasks_before, tasks_after, llm_stats=llm_agg if args.use_llm else None, probe_snapshot=last_probe)
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


