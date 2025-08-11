import os
import sys
import json
import time
import subprocess
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Prefer Streamlit secrets in cloud, then env var, then a sane hosted default
API_BASE = (
    (st.secrets.get("API_BASE") if hasattr(st, "secrets") else None)
    or os.getenv("API_BASE")
    or "https://capstone-project-api-jg3n.onrender.com"
).rstrip("/")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _list_reports() -> list[Path]:
    reports_dir = PROJECT_ROOT / "7_agent_layers"
    return sorted(reports_dir.glob("development_session_report_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_suggestions_file() -> Path | None:
    sug_dir = PROJECT_ROOT / "results" / "llm_suggestions"
    if not sug_dir.exists():
        return None
    files = sorted(sug_dir.glob("session_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_suggestions(path: Path) -> list[dict]:
    items: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            raw = obj.get("suggestions")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                items.extend(parsed)
    except Exception:
        pass
    return items


def _append_task(layer: str, text: str) -> bool:
    try:
        lid = str(layer).strip()
        folder = PROJECT_ROOT / f"7_agent_layers/LVL_{lid}"
        taskfile = folder / "tasklist.md"
        if not taskfile.exists():
            return False
        with taskfile.open("a", encoding="utf-8") as f:
            f.write(f"\n- [ ] {text}\n")
        return True
    except Exception:
        return False


def _api_get(path: str) -> tuple[int, dict | str]:
    import requests
    try:
        secret_key = (st.secrets.get("DEMO_API_KEY") if hasattr(st, "secrets") else None)
        headers = {"X-API-Key": secret_key or os.getenv("DEMO_API_KEY", "demo_key_123")}
        sess_llm_key = st.session_state.get("LLM_KEY")
        if sess_llm_key:
            headers["X-LLM-Key"] = sess_llm_key
        r = requests.get(f"{API_BASE}{path}", timeout=5, headers=headers)
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)


def _api_post(path: str, body: dict, timeout: int = 15) -> tuple[int, dict | str]:
    import requests
    try:
        secret_key = (st.secrets.get("DEMO_API_KEY") if hasattr(st, "secrets") else None)
        headers = {"X-API-Key": secret_key or os.getenv("DEMO_API_KEY", "demo_key_123")}
        sess_llm_key = st.session_state.get("LLM_KEY")
        if sess_llm_key:
            headers["X-LLM-Key"] = sess_llm_key
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout, headers=headers)
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)


def _api_status() -> dict:
    """Check API reachability and readiness.
    Returns: { api_ok: bool, ready_ok: bool, health: any, ready: any }
    """
    status: dict = {"api_ok": False, "ready_ok": False}
    hc, h = _api_get("/health")
    status["health"] = h
    status["api_ok"] = bool(hc and hc >= 200 and hc < 300)
    rc, r = _api_get("/status/ready")
    status["ready"] = r
    status["ready_ok"] = bool(rc and rc >= 200 and rc < 300 and isinstance(r, dict) and r.get("ready"))
    return status


def page_chat():
    st.header("Chat & Session Runner")
    prompt = st.text_input("Ask the agent", value="What is L4 and how does it work?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send", use_container_width=True):
            # Use new RAG chat endpoint
            code, data = _api_post("/chat/message", {"session_id": "default", "message": prompt, "top_k": 5}, timeout=60)
            if code >= 200 and code < 300 and isinstance(data, dict):
                st.success("Response:")
                st.write(data.get("answer", "(no answer)"))
            else:
                st.warning("Backend unavailable; showing placeholder response.")
                st.write(f"Echo: {prompt}")
    st.caption("Tip: Use Streamed Chat to prevent tunnel timeouts on long answers.")
    with col2:
        if st.button("Run 2-min Self-Dev Session", use_container_width=True):
            with st.status("Running self_dev_session (2 min)ā€¦", expanded=True):
                # Delegate to backend endpoint so UI works remotely too
                code, data = _api_post(
                    "/admin/run_session",
                    {
                        "duration_min": 2,
                        "sleep": 0.2,
                        "use_llm": True,
                        "preload_llm": True,
                        "max_tests_per_loop": 1,
                    },
                    timeout=200,
                )
                if code >= 200 and code < 300 and isinstance(data, dict):
                    stdout = (data.get("stdout") or "").strip()
                    stderr = (data.get("stderr") or "").strip()
                    rc = int(data.get("returncode", -1))
                    if stdout:
                        st.write(stdout[-20000:])
                    if rc == 0:
                        st.success("Session finished.")
                    else:
                        st.error(f"Session failed (rc={rc})")
                        if stderr:
                            st.code(stderr[-2000:])
                else:
                    st.error(f"Backend error: {code}")

    # Recent report
    reports = _list_reports()
    if reports:
        st.caption(f"Latest report: {reports[0].name}")
        if st.button("View chat history"):
            code, hist = _api_get("/chat/history")
            if code >= 200 and isinstance(hist, dict):
                st.code(json.dumps(hist, indent=2))

    st.divider()
    st.subheader("Streamed Chat (keeps tunnel alive)")
    streamed_q = st.text_input("Prompt (streamed)", value="Explain Layer 4 deeply")
    if st.button("Send streamed", type="primary"):
        import requests
        try:
            secret_key = (st.secrets.get("DEMO_API_KEY") if hasattr(st, "secrets") else None)
            headers = {"X-API-Key": secret_key or os.getenv("DEMO_API_KEY", "demo_key_123")}
            sess_llm_key = st.session_state.get("LLM_KEY")
            if sess_llm_key:
                headers["X-LLM-Key"] = sess_llm_key
            resp = requests.post(
                f"{API_BASE}/chat/stream",
                json={"session_id": "default", "message": streamed_q, "top_k": 5},
                headers=headers,
                stream=True,
                timeout=600,
            )
            resp.raise_for_status()
            out = ""
            ph = st.empty()
            for chunk in resp.iter_content(chunk_size=None):
                if not chunk:
                    continue
                try:
                    text = chunk.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                out += text
                ph.write(out)
        except Exception as e:
            st.error(str(e))


def page_reports():
    st.header("Self-Dev Reports")
    reports = _list_reports()
    if not reports:
        st.info("No reports found.")
        return
    names = [p.name for p in reports]
    choice = st.selectbox("Select report", names)
    sel = reports[names.index(choice)]
    st.markdown(_read_text(sel))


def page_suggestions():
    st.header("LLM Suggestions (Human-in-the-loop)")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Run suggestions job", type="primary"):
            code, data = _api_post("/selfdev/suggestions/run", {"max_files": 12})
            if code >= 200 and isinstance(data, dict):
                st.success(f"Generated {data.get('count', 0)} suggestions ā†’ {data.get('path','')}")
            else:
                st.error(f"Failed to run suggestions job: {code}")
    p = _latest_suggestions_file()
    if not p:
        st.info("No suggestion files found. Run a session to generate.")
        return
    items = _load_suggestions(p)
    if not items:
        st.info("No suggestions parsed from latest file.")
        return
    approved: list[tuple[str, str, dict]] = []
    for idx, s in enumerate(items):
        with st.expander(f"Suggestion {idx+1}"):
            layer = str(s.get("layer", "")).strip()
            path = s.get("path", "")
            rationale = s.get("rationale", "")
            steps = s.get("steps", [])
            st.write({"layer": layer, "path": path, "rationale": rationale, "steps": steps})
            new_text = st.text_input("Edit task text", value=(rationale or path or "task"), key=f"txt_{idx}")
            if st.checkbox("Approve", key=f"ok_{idx}"):
                approved.append((layer, new_text, s))
    if approved and st.button("Append approved to tasklists"):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ok = 0
        for layer, text, s in approved:
            label = f"{text} (source: LLM {ts})"
            ok += 1 if _append_task(layer, label) else 0
        st.success(f"Appended {ok} tasks to tasklists.")


def page_layers():
    st.header("Layers Overview")
    tabs = st.tabs([f"L{n}" for n in range(1, 8)])
    for i, t in enumerate(tabs, start=1):
        with t:
            layer_file = PROJECT_ROOT / f"7_agent_layers/LVL_{i}/layer{i}.md"
            task_file = PROJECT_ROOT / f"7_agent_layers/LVL_{i}/tasklist.md"
            st.subheader(f"Layer {i}")
            if layer_file.exists():
                st.markdown(_read_text(layer_file))
            else:
                st.write("No layer overview yet.")
            st.divider()
            if task_file.exists():
                st.markdown("### Tasklist")
                st.markdown(_read_text(task_file))
            else:
                st.write("No tasklist found.")


def page_mesh():
    st.header("Agent Mesh")
    mesh = PROJECT_ROOT / "7_agent_layers/mesh_map.md"
    if mesh.exists():
        st.markdown(_read_text(mesh))
    else:
        st.write("mesh_map.md not found.")


def page_health():
    st.header("Health & Metrics")

    # Overall status indicator
    stat = _api_status()
    if stat.get("ready_ok"):
        st.success("API ready")
    elif stat.get("api_ok"):
        st.warning("API reachable, warming up…")
    else:
        st.error("API unavailable")

    st.subheader("/health")
    st.code(json.dumps(stat.get("health") if isinstance(stat.get("health"), dict) else {"raw": stat.get("health")}, indent=2))

    code_m, data_m = _api_get("/performance/metrics")
    st.subheader("/performance/metrics")
    st.code(json.dumps(data_m if isinstance(data_m, dict) else {"raw": data_m}, indent=2))

    st.divider()
    st.subheader("Ollama Quick Test")
    test_prompt = st.text_input("Prompt", value="Say hello in one short sentence.")
    if st.button("Test Ollama generate"):
        code, data = _api_post("/tools/llm/generate", {"prompt": test_prompt}, timeout=60)
        if code >= 200 and code < 300:
            try:
                st.success("Generation OK")
                if isinstance(data, dict):
                    st.write(data.get("response", ""))
                else:
                    st.write(str(data))
            except Exception:
                st.write(data)
        else:
            st.error(f"Generate failed: {code}")

    st.divider()
    st.subheader("Readiness & Warmup")
    colw1, colw2 = st.columns(2)
    with colw1:
        if st.button("Check readiness"):
            st.json(_api_status())
    with colw2:
        if st.button("Warm model", type="primary"):
            c, d = _api_post("/admin/warmup", {}, timeout=180)
            if c >= 200 and isinstance(d, dict):
                if d.get("ready"):
                    st.success(f"Warmed. embed_ms={d.get('embed_ms')}, generate_ms={d.get('generate_ms')}")
                else:
                    st.warning(d)
            else:
                st.error(f"Warmup failed: {c}")

    st.divider()
    if st.button("Run local health_probe.py"):
        with st.status("Running health_probe.pyā€¦", expanded=False):
            try:
                proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "health_probe.py")],
                                      cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60)
                out = proc.stdout.strip()
                st.code(out or proc.stderr)
            except Exception as e:
                st.error(str(e))


def page_tools():
    st.header("Tools / APIs")
    # Fetch tools list on load
    code, tools_list = _api_get("/tools/list")
    if code >= 200 and code < 300 and isinstance(tools_list, dict):
        st.success("Fetched tools list")
        st.json(tools_list)
    else:
        st.info("Could not fetch tools list (requires X-API-Key). Showing static info instead.")
        st.write({
            "API_BASE": API_BASE,
            "key_endpoints": [
                "/health",
                "/performance/metrics",
                "/tools/registry/health",
                "/api/v1/tools/call",
                "/tools/llm/generate",
                "/memory/*",
            ]
        })
    code, data = _api_get("/tools/registry/health")
    st.subheader("/tools/registry/health")
    if code:
        st.code(json.dumps(data if isinstance(data, dict) else {"raw": data}, indent=2))
    else:
        st.info("Endpoint not available.")

    st.divider()
    st.subheader("Ingest Files to Memory (RAG)")
    uploaded = st.file_uploader("Upload one or more files", type=[
        "md","txt","json","csv","yaml","yml","html","pdf","docx","png","jpg","jpeg","bmp","tiff"
    ], accept_multiple_files=True)
    src = st.text_input("Source tag (optional)", value="ui-upload")
    if uploaded and st.button("Ingest", type="primary"):
        import requests
        try:
            secret_key = (st.secrets.get("DEMO_API_KEY") if hasattr(st, "secrets") else None)
            headers = {"X-API-Key": secret_key or os.getenv("DEMO_API_KEY", "demo_key_123")}
            sess_llm_key = st.session_state.get("LLM_KEY")
            files = []
            for f in uploaded:
                files.append(("files", (f.name, f.getvalue(), "application/octet-stream")))
            data = {"source": src}
            resp = requests.post(f"{API_BASE}/ingest/files", headers=headers, files=files, data=data, timeout=180)
            if resp.ok:
                st.success(f"Ingested {resp.json().get('stored',0)} files; errors={resp.json().get('errors',0)}")
                st.json(resp.json())
            else:
                st.error(f"Ingest failed: {resp.status_code}")
                st.text(resp.text)
        except Exception as e:
            st.error(str(e))


def main():
    st.set_page_config(page_title="Agent Demo UI", layout="wide")
    with st.sidebar:
        st.title("Agent Demo UI")
        st.caption(f"API_BASE: {API_BASE}")
        st.divider()
        st.subheader("LLM Key (optional)")
        current_key = st.session_state.get("LLM_KEY", "")
        new_key = st.text_input("X-LLM-Key header", value=current_key, type="password")
        colk1, colk2 = st.columns(2)
        with colk1:
            if st.button("Save Key"):
                st.session_state["LLM_KEY"] = new_key
                st.success("Saved")
        with colk2:
            if st.button("Clear Key"):
                st.session_state.pop("LLM_KEY", None)
                st.info("Cleared")
        st.divider()
        page = st.radio("Navigate", ["Chat", "Reports", "Suggestions", "Layers", "Mesh", "Health", "Tools/APIs"], index=0)

    if page == "Chat":
        page_chat()
    elif page == "Reports":
        page_reports()
    elif page == "Suggestions":
        page_suggestions()
    elif page == "Layers":
        page_layers()
    elif page == "Mesh":
        page_mesh()
    elif page == "Health":
        page_health()
    else:
        page_tools()


if __name__ == "__main__":
    main()



