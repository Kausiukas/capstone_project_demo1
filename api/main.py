import os
import json
import time
from pathlib import Path
import socket
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    # OpenAI SDK v1
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


API_KEY_ENV = os.getenv("API_KEY", "demo_key_123")


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if API_KEY_ENV and x_api_key != API_KEY_ENV:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="Capstone Demo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------------------
# Basic health and metrics
# ---------------------------
@app.get("/health")
def health(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "instance": {
            "hostname": HOSTNAME,
            "started_at": STARTED_AT,
            "image_tag": IMAGE_TAG,
            "commit_sha": COMMIT_SHA,
            "build_time": BUILD_TIME,
        },
    }


@app.get("/performance/metrics")
def performance_metrics(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {
        "overall": {"success_rate": 99.5, "avg_response_time": 12.3, "total_requests": 1234},
        "tools": {
            "ping": {"success_rate": 100.0, "avg_response_time": 5.1, "total_requests": 100},
            "list_files": {"success_rate": 99.0, "avg_response_time": 9.9, "total_requests": 50},
        },
    }


@app.get("/performance/alerts")
def performance_alerts(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"alerts": []}


@app.get("/performance/dashboard")
def performance_dashboard(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"dashboard": {"active_tools": 5, "uptime_hours": 12.5, "error_rate": 0.05}}


@app.get("/performance/health")
def performance_health(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"status": "ok", "metrics": {"cpu_usage": 12.1, "memory_usage": 43.2}}


# Tools registry
@app.get("/tools/list")
def tools_list(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"tools": ["ping", "list_files", "read_file", "get_system_status", "analyze_code"]}


@app.get("/tools/registry/health")
def tools_registry_health(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"status": "ok", "registered": 5}


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


@app.post("/api/v1/tools/call")
def tools_call(body: ToolCall, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    if body.name == "ping":
        return {
            "content": [{"type": "text", "text": "pong"}],
            "instance": {
                "hostname": HOSTNAME,
                "started_at": STARTED_AT,
                "image_tag": IMAGE_TAG,
                "commit_sha": COMMIT_SHA,
                "build_time": BUILD_TIME,
            },
        }
    return {"content": [{"type": "text", "text": f"Executed {body.name} with {body.arguments}"}]}


# ---------------------------
# Simple RAG + LLM wiring
# ---------------------------

MEMORY_PATH = Path(os.getenv("MEMORY_PATH", "./memory_store.json")).resolve()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Provider switch: "ollama" or default "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://ollama:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# Process identity
STARTED_AT = int(time.time())
HOSTNAME = os.getenv("HOSTNAME") or socket.gethostname()
IMAGE_TAG = os.getenv("IMAGE_TAG", "local")
COMMIT_SHA = os.getenv("COMMIT_SHA", "dev")
BUILD_TIME = os.getenv("BUILD_TIME", "")


def get_llm_key(x_llm_key: Optional[str] = Header(None)) -> Optional[str]:
    """Optional per-request LLM key override provided by the caller via header 'X-LLM-Key'."""
    return x_llm_key


def _ensure_memory_file() -> None:
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(json.dumps({"items": []}), encoding="utf-8")


def _load_memory() -> Dict[str, Any]:
    _ensure_memory_file()
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}


def _save_memory(data: Dict[str, Any]) -> None:
    MEMORY_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _get_openai_client(override_key: Optional[str] = None) -> Optional[OpenAI]:
    api_key = override_key or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _ollama_post(path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _ollama_get(path: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _embed_texts_openai(client: OpenAI, texts: List[str]) -> List[List[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]


def _embed_texts_ollama(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in texts:
        data = _ollama_post(
            "/api/embeddings",
            {"model": OLLAMA_EMBED_MODEL, "prompt": t},
            timeout=300,
        )
        # Ollama returns {"embedding": [...]}
        vec = data.get("embedding") or data.get("embeddings")
        if isinstance(vec, list):
            vectors.append(vec)
        else:
            vectors.append([0.0] * EMBED_DIM)
    return vectors


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embed_texts(client: Optional[OpenAI], texts: List[str]) -> List[List[float]]:
    if LLM_PROVIDER == "ollama":
        return _embed_texts_ollama(texts)
    if client is None:
        return []
    return _embed_texts_openai(client, texts)


def _retrieve(client: Optional[OpenAI], query: str, k: int = 5) -> List[Dict[str, Any]]:
    mem = _load_memory()
    items: List[Dict[str, Any]] = mem.get("items", [])
    if not items:
        return []
    embs = _embed_texts(client, [query])
    if not embs:
        return []
    q_emb = np.array(embs[0], dtype=np.float32)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in items:
        emb = np.array(item.get("embedding", []), dtype=np.float32)
        scored.append((_cosine_sim(q_emb, emb), item))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [it for _, it in scored[:k]]


# LLM raw generate
class GenerateBody(BaseModel):
    prompt: str


@app.post("/tools/llm/generate")
def llm_generate(body: GenerateBody, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    if LLM_PROVIDER == "ollama":
        try:
            data = _ollama_post(
                "/api/generate",
                {"model": OLLAMA_CHAT_MODEL, "prompt": body.prompt, "stream": False},
                timeout=600,
            )
            return {"response": (data.get("response") or "").strip()}
        except Exception as e:
            return {"response": f"[ollama error] {e}"}
    client = _get_openai_client(llm_key)
    if client is None:
        return {"response": f"[mock] You said: {body.prompt}"}
    msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": body.prompt},
    ]
    res = client.chat.completions.create(model=CHAT_MODEL, messages=msg)
    text = (res.choices[0].message.content or "").strip()
    return {"response": text}


# Chat and session
class ChatMessage(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


@app.post("/chat/message")
def chat_message(body: ChatMessage, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    # Retrieve (provider-agnostic embeddings)
    client = _get_openai_client(llm_key) if LLM_PROVIDER != "ollama" else None
    contexts = _retrieve(client, body.message, k=max(1, min(10, body.top_k)))
    docs = "\n\n".join([f"[Doc {i+1}]\n" + (c.get("text") or "") for i, c in enumerate(contexts)])

    system = (
        "You are the agent brain with access to retrieved context from prior sessions. "
        "Use the provided documents if relevant. If not enough information is present, say so briefly."
    )
    user_msg = (
        f"User question:\n{body.message}\n\n"
        f"Retrieved context:\n{docs if docs else '(no context)'}"
    )

    if LLM_PROVIDER == "ollama":
        try:
            data = _ollama_post(
                "/api/generate",
                {
                    "model": OLLAMA_CHAT_MODEL,
                    "prompt": system + "\n\n" + user_msg,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=600,
            )
            answer = (data.get("response") or "").strip()
            return {"answer": answer, "used_docs": len(contexts)}
        except Exception as e:
            return {"answer": f"[ollama error] {e}"}

    # OpenAI path
    client = _get_openai_client(llm_key)
    if client is None:
        return {"answer": f"[mock] LLM answer to: {body.message}"}
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.3,
    )
    answer = (res.choices[0].message.content or "").strip()
    return {"answer": answer, "used_docs": len(contexts)}


@app.post("/chat/stream")
def chat_stream(body: ChatMessage, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)):
    """Stream chat answer with periodic keepalives so tunnels/proxies don't time out.
    Sends small whitespace heartbeats every ~8 seconds and yields partial tokens
    as available. Client should treat it as text/event-stream or chunked text.
    """
    client = _get_openai_client(llm_key) if LLM_PROVIDER != "ollama" else None
    contexts = _retrieve(client, body.message, k=max(1, min(10, body.top_k)))
    docs = "\n\n".join([f"[Doc {i+1}]\n" + (c.get("text") or "") for i, c in enumerate(contexts)])
    system = (
        "You are the agent brain with access to retrieved context from prior sessions. "
        "Use the provided documents if relevant. If not enough information is present, say so briefly."
    )
    user_msg = (
        f"User question:\n{body.message}\n\n" f"Retrieved context:\n{docs if docs else '(no context)'}"
    )

    def _stream_ollama():
        import datetime
        last = time.time()
        try:
            # Use generate streaming
            url = f"{OLLAMA_BASE}/api/generate"
            with requests.post(
                url,
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "prompt": system + "\n\n" + user_msg,
                    "stream": True,
                    "options": {"temperature": 0.3},
                },
                stream=True,
                timeout=600,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    now = time.time()
                    if not line:
                        # emit keepalive every 8s
                        if now - last > 8:
                            last = now
                            yield " \n"
                        continue
                    # Ollama streams JSON per line {response: '...', done: bool}
                    try:
                        obj = json.loads(line)
                        chunk = (obj.get("response") or "")
                        if chunk:
                            yield chunk
                            last = now
                        if obj.get("done"):
                            break
                    except Exception:
                        yield " \n"
        except Exception as e:
            yield f"[stream error] {e}"

    def _stream_openai():
        last = time.time()
        if client is None:
            # mock stream with small chunks
            txt = f"[mock] LLM answer to: {body.message}"
            for ch in txt:
                yield ch
                time.sleep(0.01)
            return
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                temperature=0.3,
                stream=True,
            )
            for ev in stream:
                now = time.time()
                if hasattr(ev, "choices") and ev.choices:
                    delta = ev.choices[0].delta.content or ""
                    if delta:
                        yield delta
                        last = now
                if now - last > 8:
                    yield " \n"
                    last = now
        except Exception as e:
            yield f"[stream error] {e}"

    generator = _stream_ollama if LLM_PROVIDER == "ollama" else _stream_openai
    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")

class RunSessionBody(BaseModel):
    duration_min: int = 2
    sleep: float = 0.2
    use_llm: bool = True
    preload_llm: bool = True
    max_tests_per_loop: int = 1


@app.post("/admin/run_session")
def run_session(_: RunSessionBody, __: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"returncode": 0, "stdout": "[mock] session completed", "stderr": ""}


# ---------------------------
# Readiness + Warmup
# ---------------------------

class WarmupResult(BaseModel):
    provider: str
    embed_ms: Optional[float] = None
    generate_ms: Optional[float] = None
    ready: bool = False


@app.get("/status/ready")
def status_ready(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    ready = False
    details: Dict[str, Any] = {"provider": LLM_PROVIDER}
    if LLM_PROVIDER == "ollama":
        try:
            _ = _ollama_get("/api/tags", timeout=5)
            details["ollama"] = "up"
            ready = True
        except Exception as e:
            details["ollama_error"] = str(e)
            ready = False
    else:
        details["has_openai_key"] = bool(os.getenv("OPENAI_API_KEY"))
        ready = bool(os.getenv("OPENAI_API_KEY"))
    return {"ready": ready, "details": details}


@app.post("/admin/warmup", response_model=WarmupResult)
def admin_warmup(_: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> WarmupResult:
    embed_ms: Optional[float] = None
    generate_ms: Optional[float] = None
    if LLM_PROVIDER == "ollama":
        try:
            t0 = time.time()
            _ = _ollama_post("/api/embeddings", {"model": OLLAMA_EMBED_MODEL, "prompt": "warmup"}, timeout=180)
            embed_ms = (time.time() - t0) * 1000.0
        except Exception:
            embed_ms = None
        try:
            t0 = time.time()
            _ = _ollama_post("/api/generate", {"model": OLLAMA_CHAT_MODEL, "prompt": "hi", "stream": False}, timeout=300)
            generate_ms = (time.time() - t0) * 1000.0
        except Exception:
            generate_ms = None
        ready = (embed_ms is not None) and (generate_ms is not None)
        return WarmupResult(provider=LLM_PROVIDER, embed_ms=embed_ms, generate_ms=generate_ms, ready=ready)

    # OpenAI path
    client = _get_openai_client(llm_key)
    if client is None:
        return WarmupResult(provider=LLM_PROVIDER, embed_ms=None, generate_ms=None, ready=False)
    try:
        t0 = time.time()
        _ = _embed_texts_openai(client, ["warmup text"])
        embed_ms = (time.time() - t0) * 1000.0
    except Exception:
        embed_ms = None
    try:
        t0 = time.time()
        _ = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": "hi"}])
        generate_ms = (time.time() - t0) * 1000.0
    except Exception:
        generate_ms = None
    ready = (embed_ms is not None) and (generate_ms is not None)
    return WarmupResult(provider=LLM_PROVIDER, embed_ms=embed_ms, generate_ms=generate_ms, ready=ready)

# Memory store/query (JSON + OpenAI embeddings)
class StoreBody(BaseModel):
    user_input: str
    response: Optional[str] = None
    response_b64: Optional[str] = None


@app.post("/memory/store")
def memory_store(body: StoreBody, __: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    raw_resp = body.response or ""
    if (not raw_resp) and body.response_b64:
        try:
            import base64
            raw_resp = base64.b64decode(body.response_b64).decode("utf-8", errors="ignore")
        except Exception:
            raw_resp = ""
    text = (body.user_input or "").strip() + "\n" + raw_resp.strip()
    client = _get_openai_client(llm_key) if LLM_PROVIDER != "ollama" else None
    vecs = _embed_texts(client, [text])
    embedded = bool(vecs)
    vec = vecs[0] if embedded else []
    mem = _load_memory()
    mem.setdefault("items", []).append(
        {"id": f"m_{int(time.time()*1000)}", "text": text, "embedding": vec, "ts": int(time.time())}
    )
    _save_memory(mem)
    return {"stored": True, "embedded": embedded}


@app.get("/memory/query")
def memory_query(q: str, k: int = 5, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    client = _get_openai_client(llm_key) if LLM_PROVIDER != "ollama" else None
    items = _retrieve(client, q, k=max(1, min(20, k)))
    return {"snippets": [{"preview": it.get("text", "")[:800]} for it in items]}


# Content preview (mock)
@app.get("/preview/file")
def preview_file(file_path: str, preview_type: Optional[str] = None, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    content = f"[mock preview] {file_path} ({preview_type or 'auto'})"
    return {"file_type": preview_type or "text", "language": "markdown", "content_length": len(content), "content": content, "metadata": {"source_type": "mock"}, "preview_html": f"<pre>{content}</pre>"}


@app.get("/preview/analyze")
def preview_analyze(file_path: str, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {
        "preview_type": "markdown",
        "language": "markdown",
        "supported": True,
        "source_type": "mock",
        "file_extension": ".md",
        "exists": True,
        "capabilities": {"syntax_highlighting": True, "markdown_rendering": True},
    }


@app.get("/preview/supported-types")
def preview_supported_types(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"supported_types": {"code": {"extensions": [".py", ".js"], "mime_types": ["text/plain"]}, "markdown": {"extensions": [".md"], "mime_types": ["text/markdown"]}}}


class BatchBody(BaseModel):
    file_paths: List[str]


@app.post("/preview/batch")
def preview_batch(body: BatchBody, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    results = []
    for p in body.file_paths:
        content = f"[mock batch preview] {p}"
        results.append({"file_path": p, "success": True, "file_type": "text", "language": "markdown", "content_length": len(content), "preview_html": f"<pre>{content}</pre>"})
    return {"successful_previews": len(results), "total_files": len(body.file_paths), "results": results}


# ---------------------------
# Component status snapshot
# ---------------------------

@app.get("/status/components")
def status_components(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    # API basics
    components["api"] = {
        "service": "capstone-demo-api",
        "llm_provider": LLM_PROVIDER,
        "hostname": HOSTNAME,
        "started_at": STARTED_AT,
        "image_tag": IMAGE_TAG,
        "commit_sha": COMMIT_SHA,
        "build_time": BUILD_TIME,
    }
    # Endpoints availability
    components["endpoints"] = {
        "status_ready": True,
        "admin_warmup": True,
        "chat_stream": True,
        "ingest_files": True,
    }
    # Memory
    try:
        mem = _load_memory()
        components["memory"] = {"path": str(MEMORY_PATH), "items": len(mem.get("items", []))}
    except Exception as e:
        components["memory"] = {"error": str(e)}
    # LLM provider quick signal
    if LLM_PROVIDER == "ollama":
        try:
            _ = _ollama_get("/api/tags", timeout=5)
            components["llm"] = {"ollama": "up", "base": OLLAMA_BASE, "chat_model": OLLAMA_CHAT_MODEL, "embed_model": OLLAMA_EMBED_MODEL}
        except Exception as e:
            components["llm"] = {"ollama": "down", "error": str(e)}
    else:
        components["llm"] = {"openai_key": bool(os.getenv("OPENAI_API_KEY")), "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}
    return components


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "capstone-demo-api", "status": "ok"}


# ---------------------------
# Ingestion utilities
# ---------------------------

def _extract_text_from_bytes(content: bytes, filename: str) -> str:
    name = (filename or "").lower()
    try:
        if name.endswith(('.md', '.txt', '.json', '.csv', '.log', '.yaml', '.yml', '.py', '.js', '.ts', '.html', '.htm')):
            return content.decode('utf-8', errors='ignore')
        if name.endswith('.pdf'):
            from io import BytesIO
            from pdfminer.high_level import extract_text
            return extract_text(BytesIO(content)) or ""
        if name.endswith('.docx'):
            from io import BytesIO
            import docx
            doc = docx.Document(BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
        if name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                import pytesseract  # optional if available
                from PIL import Image
                from io import BytesIO
                img = Image.open(BytesIO(content))
                return pytesseract.image_to_string(img) or ""
            except Exception:
                return ""
    except Exception:
        return ""
    # Fallback
    return content.decode('utf-8', errors='ignore')


class IngestResponse(BaseModel):
    stored: int
    errors: int
    details: List[Dict[str, Any]]


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    source: Optional[str] = Form(None),
    _: None = Depends(require_api_key),
    llm_key: Optional[str] = Depends(get_llm_key),
):
    stored = 0
    errs = 0
    details: List[Dict[str, Any]] = []
    for uf in files:
        try:
            data = await uf.read()
            text = _extract_text_from_bytes(data, uf.filename or "file")
            if not text.strip():
                details.append({"file": uf.filename, "ok": False, "reason": "empty"})
                errs += 1
                continue
            body = StoreBody(user_input=f"upload:{uf.filename} {source or ''}", response=text)
            memory_store(body, None, llm_key)  # reuse in-process
            details.append({"file": uf.filename, "ok": True, "chars": len(text)})
            stored += 1
        except Exception as e:
            errs += 1
            details.append({"file": uf.filename, "ok": False, "error": str(e)})
    return IngestResponse(stored=stored, errors=errs, details=details)



