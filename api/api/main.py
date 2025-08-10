import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
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
    return {"status": "ok"}


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
        return {"content": [{"type": "text", "text": "pong"}]}
    return {"content": [{"type": "text", "text": f"Executed {body.name} with {body.arguments}"}]}


# ---------------------------
# Simple RAG + LLM wiring
# ---------------------------

MEMORY_PATH = Path(os.getenv("MEMORY_PATH", "./memory_store.json")).resolve()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


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


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _retrieve(client: OpenAI, query: str, k: int = 5) -> List[Dict[str, Any]]:
    mem = _load_memory()
    items: List[Dict[str, Any]] = mem.get("items", [])
    if not items:
        return []
    q_emb = np.array(_embed_texts(client, [query])[0], dtype=np.float32)
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
def llm_generate(body: GenerateBody, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    client = _get_openai_client()
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
def chat_message(body: ChatMessage, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    client = _get_openai_client()
    if client is None:
        return {"answer": f"[mock] LLM answer to: {body.message}"}

    # Retrieve
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
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.3,
    )
    answer = (res.choices[0].message.content or "").strip()
    return {"answer": answer, "used_docs": len(contexts)}


class RunSessionBody(BaseModel):
    duration_min: int = 2
    sleep: float = 0.2
    use_llm: bool = True
    preload_llm: bool = True
    max_tests_per_loop: int = 1


@app.post("/admin/run_session")
def run_session(_: RunSessionBody, __: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"returncode": 0, "stdout": "[mock] session completed", "stderr": ""}


# Memory store/query (JSON + OpenAI embeddings)
class StoreBody(BaseModel):
    user_input: str
    response: str


@app.post("/memory/store")
def memory_store(body: StoreBody, __: None = Depends(require_api_key)) -> Dict[str, Any]:
    client = _get_openai_client()
    if client is None:
        # Still store raw text without embeddings for future runs
        mem = _load_memory()
        mem.setdefault("items", []).append(
            {
                "id": f"m_{int(time.time()*1000)}",
                "text": (body.user_input or "") + "\n" + (body.response or ""),
                "embedding": [],
                "ts": int(time.time()),
            }
        )
        _save_memory(mem)
        return {"stored": True, "embedded": False}

    text = (body.user_input or "").strip() + "\n" + (body.response or "").strip()
    vec = _embed_texts(client, [text])[0]
    mem = _load_memory()
    mem.setdefault("items", []).append(
        {"id": f"m_{int(time.time()*1000)}", "text": text, "embedding": vec, "ts": int(time.time())}
    )
    _save_memory(mem)
    return {"stored": True, "embedded": True}


@app.get("/memory/query")
def memory_query(q: str, k: int = 5, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    client = _get_openai_client()
    if client is None:
        return {"snippets": [{"preview": f"[mock] result for '{q}' #{i+1}"} for i in range(k)]}
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


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "capstone-demo-api", "status": "ok"}


