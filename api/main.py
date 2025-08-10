import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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


# Health and performance
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


# LLM endpoints (mock)
class GenerateBody(BaseModel):
    prompt: str


@app.post("/tools/llm/generate")
def llm_generate(body: GenerateBody, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"response": f"[mock] You said: {body.prompt}"}


# Chat and session
class ChatMessage(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


@app.post("/chat/message")
def chat_message(body: ChatMessage, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"answer": f"[mock] LLM answer to: {body.message}"}


class RunSessionBody(BaseModel):
    duration_min: int = 2
    sleep: float = 0.2
    use_llm: bool = True
    preload_llm: bool = True
    max_tests_per_loop: int = 1


@app.post("/admin/run_session")
def run_session(_: RunSessionBody, __: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"returncode": 0, "stdout": "[mock] session completed", "stderr": ""}


# Memory (mock)
class StoreBody(BaseModel):
    user_input: str
    response: str


@app.post("/memory/store")
def memory_store(_: StoreBody, __: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"stored": True}


@app.get("/memory/query")
def memory_query(q: str, k: int = 5, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"snippets": [{"preview": f"[mock] result for '{q}' #{i+1}"} for i in range(k)]}


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


