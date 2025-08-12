import os
import json
import time
from pathlib import Path
import socket
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import platform
import json as _json
import asyncio
from datetime import datetime

import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    # OpenAI SDK v1
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Import PostgreSQL memory system
try:
    from src.layers.memory_system import MemorySystem
    POSTGRES_MEMORY_AVAILABLE = True
except ImportError:
    POSTGRES_MEMORY_AVAILABLE = False
    MemorySystem = None

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


# Identity headers middleware
@app.middleware("http")
async def add_identity_headers(request, call_next):
    response = await call_next(request)
    try:
        response.headers["X-Api-Hostname"] = HOSTNAME
        response.headers["X-Api-Started-At"] = str(STARTED_AT)
        response.headers["X-Api-Image-Tag"] = IMAGE_TAG
        response.headers["X-Api-Commit"] = COMMIT_SHA
        response.headers["X-Api-Instance-Id"] = INSTANCE_ID
    except Exception:
        pass
    return response


# ---------------------------
# Basic health and metrics
# ---------------------------
def _load_identity_from_file() -> Dict[str, Any]:
    """Optionally load identity from a JSON file specified by IDENTITY_FILE or ./identity.json.
    Values in this file augment environment-derived identity.
    """
    locations = []
    env_path = os.getenv("IDENTITY_FILE")
    if env_path:
        locations.append(Path(env_path))
    # next to this module
    locations.append(Path(__file__).resolve().parent / "identity.json")
    # project root (two levels up from api/main.py)
    locations.append(Path(__file__).resolve().parents[1] / "identity.json")
    for p in locations:
        try:
            if p.exists():
                return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


_IDENTITY_FILE_DATA = _load_identity_from_file()


def build_identity() -> Dict[str, Any]:
    """Build a standardized identity object for the API instance.

    Priority order per field: file override → environment → computed defaults.
    """
    service = _IDENTITY_FILE_DATA.get("service") or os.getenv("SERVICE_NAME", "capstone-demo-api")
    version = _IDENTITY_FILE_DATA.get("version") or os.getenv("SERVICE_VERSION", "0.1.0")
    # Select a human-visible identificator if provided, else use instance hash
    identificator = (
        _IDENTITY_FILE_DATA.get("identificator")
        or os.getenv("API_IDENTIFICATOR")
        or INSTANCE_ID
    )
    identity: Dict[str, Any] = {
        "service": service,
        "version": version,
        "hostname": HOSTNAME,
        "started_at": STARTED_AT,
        "image_tag": _IDENTITY_FILE_DATA.get("image_tag") or IMAGE_TAG,
        "commit_sha": _IDENTITY_FILE_DATA.get("commit_sha") or COMMIT_SHA,
        "build_time": _IDENTITY_FILE_DATA.get("build_time") or BUILD_TIME,
        "instance_id": INSTANCE_ID,
        "identificator": identificator,
        "python": platform.python_version(),
        "pid": os.getpid(),
    }
    # Allow custom fields from file under "extras"
    extras = _IDENTITY_FILE_DATA.get("extras")
    if isinstance(extras, dict):
        identity["extras"] = extras
    return identity


@app.get("/identity")
def identity(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return build_identity()


@app.get("/health")
def health(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"status": "ok", "instance": build_identity()}


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
    # Reverted to canonical tool names (including get_system_status)
    return {"tools": ["ping", "list_files", "read_file", "analyze_code", "get_system_status"]}


@app.get("/tools/registry/health")
def tools_registry_health(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {"status": "ok", "registered": 5}


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


@app.post("/api/v1/tools/call")
def tools_call(body: ToolCall, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    if body.name == "ping":
        # Include lightweight components so UI can show instance + LLM + Memory + Endpoints
        components = _components_snapshot()
        return {
            "content": [{"type": "text", "text": "pong"}],
            "instance": build_identity(),
            "llm": components.get("llm", {}),
            "memory": components.get("memory", {}),
            "endpoints": components.get("endpoints", {}),
        }
    if body.name == "list_files":
        try:
            limit = int(body.arguments.get("limit", 20))
        except Exception:
            limit = 20
        # Reuse internal helper
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            data = loop.run_until_complete(_list_documents(limit=limit))
        finally:
            loop.close()
        return {"tool": "list_files", "data": data}
    if body.name == "read_file":
        # Expect argument: id
        doc_id = body.arguments.get("id")
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            data: Dict[str, Any]
            data = {"success": False, "error": "invalid_id"}
            if doc_id is not None:
                try:
                    ms = loop.run_until_complete(_get_simple_memory_system())
                    if ms:
                        doc = loop.run_until_complete(ms.get_document_by_id(int(doc_id)))
                        if doc:
                            data = {"success": True, "document": doc, "type": "postgresql_pgvector"}
                        else:
                            data = {"success": False, "error": "not_found"}
                    else:
                        # Fallback to JSON by index
                        mem = _load_memory()
                        items = mem.get("items", [])
                        idx = max(0, min(len(items) - 1, int(doc_id) - 1))
                        if 0 <= idx < len(items):
                            it = items[idx]
                            data = {"success": True, "document": {"id": str(doc_id), "source": "json_file", "content": it.get("response", "")}, "type": "json_file"}
                        else:
                            data = {"success": False, "error": "not_found"}
                except Exception as e:
                    data = {"success": False, "error": str(e)}
        finally:
            loop.close()
        return {"tool": "read_file", "data": data}
    if body.name == "analyze_code":
        # Use preview/analyze logic (mock code analyzer)
        try:
            file_path = str(body.arguments.get("file_path", ""))
            res = preview_analyze(file_path, None)  # type: ignore[arg-type]
            return {"tool": "analyze_code", "data": res}
        except Exception as e:
            return {"tool": "analyze_code", "error": str(e)}
    if body.name == "get_system_status":
        # Aggregate basic status for UI diagnostics
        try:
            return {
                "tool": "get_system_status",
                "instance": build_identity(),
                "components": _components_snapshot(),
            }
        except Exception as e:
            return {"tool": "get_system_status", "error": str(e)}
    return {"content": [{"type": "text", "text": f"Executed {body.name} with {body.arguments}"}]}


# ---------------------------
# Simple RAG + LLM wiring
# ---------------------------

# Memory system configuration
MEMORY_PATH = Path(os.getenv("MEMORY_PATH", "./memory_store.json")).resolve()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Provider switch: "ollama" or default "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://lfc-ollama:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# Simple PostgreSQL memory system
import asyncpg
from typing import List, Dict, Any

# Import Interaction class
try:
    from src.layers.memory_system import Interaction
except ImportError:
    # Fallback if import fails
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class Interaction:
        user_input: str
        response: str
        tool_calls: List[Any]
        metadata: Dict[str, Any]
        timestamp: datetime
        source: str = "unknown"

class SimplePostgreSQLMemory:
    """Simple PostgreSQL memory system for direct database access"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        try:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=5,
                command_timeout=15
            )
            return True
        except Exception as e:
            print(f"Failed to initialize PostgreSQL pool: {e}")
            return False
    
    async def query_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based search (fallback when embeddings not available)"""
        try:
            if not self._pool:
                await self.initialize()
            
            async with self._pool.acquire() as conn:
                # Simple text search for now
                results = await conn.fetch(
                    """
                    SELECT id, source, content, created_at 
                    FROM vector_store 
                    WHERE content ILIKE $1 OR source ILIKE $1
                    ORDER BY created_at DESC 
                    LIMIT $2
                    """,
                    f"%{query}%", k
                )
                
                return [
                    {
                        "id": str(r["id"]),
                        "source": r["source"],
                        "content": r["content"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"PostgreSQL query failed: {e}")
            return []
    
    async def get_document_count(self) -> int:
        """Get total document count"""
        try:
            if not self._pool:
                await self.initialize()
            
            async with self._pool.acquire() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM vector_store")
        except Exception as e:
            print(f"Failed to get document count: {e}")
            return 0
    
    async def get_sample_documents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample documents for preview"""
        try:
            if not self._pool:
                await self.initialize()
            
            async with self._pool.acquire() as conn:
                results = await conn.fetch(
                    """
                    SELECT id, source, content, created_at 
                    FROM vector_store 
                    ORDER BY created_at DESC 
                    LIMIT $1
                    """,
                    limit
                )
                
                return [
                    {
                        "id": str(r["id"]),
                        "source": r["source"],
                        "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"Failed to get sample documents: {e}")
            return []

    async def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Fetch full document by id."""
        try:
            if not self._pool:
                await self.initialize()
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, source, content, created_at
                    FROM vector_store
                    WHERE id = $1
                    """,
                    doc_id,
                )
                if not row:
                    return None
                return {
                    "id": str(row["id"]),
                    "source": row["source"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
        except Exception as e:
            print(f"Failed to get document by id: {e}")
            return None
    
    async def store_interaction(self, interaction: Interaction, embedding: List[float]) -> Dict[str, Any]:
        """Store interaction with embedding in PostgreSQL"""
        try:
            if not self._pool:
                await self.initialize()
            
            async with self._pool.acquire() as conn:
                # Store the interaction in vector_store table
                new_id = await conn.fetchval(
                    """
                    INSERT INTO vector_store (source, content, created_at)
                    VALUES ($1, $2, $3)
                    RETURNING id
                    """,
                    getattr(interaction, 'source', 'unknown'),
                    f"{interaction.user_input}\n{interaction.response}",
                    datetime.now()
                )
                return {"success": True, "record_id": int(new_id)}
        except Exception as e:
            print(f"Failed to store interaction in PostgreSQL: {e}")
            return {"success": False, "error": str(e)}

# Global memory system instance
_simple_memory_system: Optional[SimplePostgreSQLMemory] = None
_memory_system_lock = asyncio.Lock()

async def _get_simple_memory_system() -> Optional[SimplePostgreSQLMemory]:
    """Get or create simple PostgreSQL memory system"""
    global _simple_memory_system
    
    if _simple_memory_system is None:
        async with _memory_system_lock:
            if _simple_memory_system is None:
                try:
                    _simple_memory_system = SimplePostgreSQLMemory(
                        connection_string=os.getenv("DATABASE_URL")
                    )
                    await _simple_memory_system.initialize()
                except Exception as e:
                    print(f"Warning: Failed to initialize simple PostgreSQL memory system: {e}")
                    return None
    
    return _simple_memory_system

def _get_memory_system_sync() -> Optional[SimplePostgreSQLMemory]:
    """Synchronous wrapper for getting memory system"""
    try:
        # Check if we're in an async context
        import asyncio
        try:
            asyncio.get_running_loop()
            # We're in async context, return None to avoid conflicts
            return None
        except RuntimeError:
            # No event loop, safe to create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_get_simple_memory_system())
                return result
            finally:
                loop.close()
    except Exception as e:
        print(f"Failed to get memory system synchronously: {e}")
        return None

# ---------------------------
# Tool helpers
# ---------------------------
async def _list_documents(limit: int = 20) -> Dict[str, Any]:
    """Return list of documents from PostgreSQL if available, else JSON fallback."""
    try:
        mem_system = await _get_simple_memory_system()
        if mem_system:
            docs = await mem_system.get_sample_documents(limit=limit)
            total = await mem_system.get_document_count()
            return {
                "success": True,
                "type": "postgresql_pgvector",
                "total_documents": total,
                "documents": docs,
            }
    except Exception:
        # Fall back below
        pass

    # Fallback to JSON memory file
    mem = _load_memory()
    items = mem.get("items", [])
    documents: List[Dict[str, Any]] = []
    for idx, item in enumerate(items[:limit]):
        documents.append(
            {
                "id": str(idx + 1),
                "source": "json_file",
                "content": (item.get("response") or "")[:200],
                "created_at": None,
            }
        )
    return {
        "success": True,
        "type": "json_file",
        "total_documents": len(items),
        "documents": documents,
    }

async def _handle_chat_tools(message: str) -> Optional[Dict[str, Any]]:
    """Very lightweight tool router for chat messages.

    Triggers:
    - Explicit command: !list_docs [limit]
    - Heuristic: user asks to list/what files/documents present
    Returns a dict with keys: tool, text, data; or None if no tool is used.
    """
    msg = (message or "").lower()

    # Explicit command parsing
    if msg.startswith("!list_docs") or any(
        phrase in msg for phrase in [
            "what files are present",
            "what documents are present",
            "what files do we have",
            "what documents do we have",
            "list files",
            "list documents",
        ]
    ):
        # Optional limit parsing for explicit command: !list_docs 50
        parts = msg.split()
        limit = 20
        if len(parts) >= 2 and parts[0] == "!list_docs":
            try:
                limit = max(1, min(200, int(parts[1])))
            except Exception:
                limit = 20

        data = await _list_documents(limit=limit)
        docs = data.get("documents", [])
        lines: List[str] = []
        for d in docs:
            lines.append(
                f"- id {d.get('id')}, source {d.get('source')}, created_at {d.get('created_at')}"
            )
        text = (
            "Here are the documents currently in memory (top "
            f"{len(docs)}):\n" + ("\n".join(lines) if lines else "(none)")
        )
        return {"tool": "list_documents", "text": text, "data": data}

    return None

# Process identity
STARTED_AT = int(time.time())
HOSTNAME = os.getenv("HOSTNAME") or socket.gethostname()
IMAGE_TAG = os.getenv("IMAGE_TAG", "local")
COMMIT_SHA = os.getenv("COMMIT_SHA", "dev")
BUILD_TIME = os.getenv("BUILD_TIME", "")
INSTANCE_ID = hashlib.sha1(
    f"{HOSTNAME}|{STARTED_AT}|{IMAGE_TAG}|{COMMIT_SHA}|{BUILD_TIME}".encode("utf-8", errors="ignore")
).hexdigest()[:16]


def get_llm_key(x_llm_key: Optional[str] = Header(None)) -> Optional[str]:
    """Optional per-request LLM key override provided by the caller via header 'X-LLM-Key'."""
    return x_llm_key


def _ensure_memory_file() -> None:
    """Fallback to JSON file if PostgreSQL not available"""
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(json.dumps({"items": []}), encoding="utf-8")


async def _load_memory_async() -> Dict[str, Any]:
    """Load memory - prefer PostgreSQL, fallback to JSON file"""
    # Try PostgreSQL first
    try:
        mem_system = await _get_simple_memory_system()
        if mem_system:
            doc_count = await mem_system.get_document_count()
            return {
                "type": "postgresql_pgvector",
                "database_url": os.getenv("DATABASE_URL", "not_set"),
                "embedding_dimension": EMBED_DIM,
                "available": True,
                "document_count": doc_count
            }
    except Exception as e:
        print(f"PostgreSQL memory system failed: {e}")
    
    # Fallback to JSON file
    _ensure_memory_file()
    try:
        return {
            "type": "json_file",
            "path": str(MEMORY_PATH),
            "items": json.loads(MEMORY_PATH.read_text(encoding="utf-8")).get("items", [])
        }
    except Exception:
        return {"type": "json_file", "path": str(MEMORY_PATH), "items": []}

def _load_memory() -> Dict[str, Any]:
    """Synchronous wrapper for memory loading"""
    try:
        # Try to get async result in sync context
        import asyncio
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use run_until_complete
            return {
                "type": "postgresql_pgvector",
                "database_url": os.getenv("DATABASE_URL", "not_set"),
                "embedding_dimension": EMBED_DIM,
                "available": True,
                "document_count": 0  # Will be updated by async calls
            }
        except RuntimeError:
            # No event loop running, safe to create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_load_memory_async())
                return result
            finally:
                loop.close()
    except Exception as e:
        print(f"Memory loading failed: {e}")
    
    # Fallback to JSON file
    _ensure_memory_file()
    try:
        return {
            "type": "json_file",
            "path": str(MEMORY_PATH),
            "items": json.loads(MEMORY_PATH.read_text(encoding="utf-8")).get("items", [])
        }
    except Exception:
        return {"type": "json_file", "path": str(MEMORY_PATH), "items": []}


def _save_memory(data: Dict[str, Any]) -> None:
    """Save memory - prefer PostgreSQL, fallback to JSON file"""
    # Try PostgreSQL first
    try:
        mem_system = _get_memory_system_sync()
        if mem_system:
            try:
                # PostgreSQL memory is handled by the MemorySystem class
                return
            except Exception:
                pass
    except Exception:
        pass
    
    # Fallback to JSON file
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
async def chat_message(body: ChatMessage, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    # Tool use: allow explicit or inferred tool invocation
    tool_reply = await _handle_chat_tools(body.message)
    if tool_reply is not None:
        # Return tool result directly, still include used_docs=0 for UX consistency
        return {"answer": tool_reply["text"], "tool": tool_reply["tool"], "data": tool_reply["data"], "used_docs": 0}

    # Try to retrieve context using PostgreSQL memory system first
    contexts = []
    try:
        mem_system = await _get_simple_memory_system()
        if mem_system:
            # Use PostgreSQL memory system
            if LLM_PROVIDER == "ollama":
                try:
                    embed_data = _ollama_post(
                        "/api/embeddings",
                        {"model": OLLAMA_EMBED_MODEL, "prompt": body.message},
                        timeout=300
                    )
                    query_embedding = embed_data.get("embedding") or embed_data.get("embeddings", [])
                except Exception as e:
                    return {"answer": f"Embedding generation failed: {e}"}
            else:
                # OpenAI embedding
                client = _get_openai_client(llm_key)
                if client:
                    try:
                        embed_res = client.embeddings.create(
                            model=EMBED_MODEL,
                            input=body.message
                        )
                        query_embedding = embed_res.data[0].embedding
                    except Exception as e:
                        return {"answer": f"OpenAI embedding failed: {e}"}
                else:
                    query_embedding = []
            
            if query_embedding:
                # Query PostgreSQL
                try:
                    result = await mem_system.query_similar(body.message, query_embedding, k=body.top_k)
                    if result.get("success"):
                        contexts = result.get("snippets", [])
                except Exception as e:
                    # Fall back to old method
                    pass
    except Exception:
        # Fall back to old method
        pass
    
    # Fallback to old retrieval method if PostgreSQL failed
    if not contexts:
        client = _get_openai_client(llm_key) if LLM_PROVIDER != "ollama" else None
        contexts = _retrieve(client, body.message, k=max(1, min(10, body.top_k)))
    
    # Format context documents
    if contexts and hasattr(contexts[0], 'get'):
        # PostgreSQL results
        docs = "\n\n".join([f"[Doc {i+1}]\n" + (c.get("preview", "") or "") for i, c in enumerate(contexts)])
    else:
        # Old format results
        docs = "\n\n".join([f"[Doc {i+1}]\n" + (c.get("text", "") or "") for i, c in enumerate(contexts)])

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
async def chat_stream(body: ChatMessage, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)):
    """Stream chat answer with periodic keepalives so tunnels/proxies don't time out.
    Sends small whitespace heartbeats every ~8 seconds and yields partial tokens
    as available. Client should treat it as text/event-stream or chunked text.
    """
    # Try to retrieve context using PostgreSQL memory system first
    contexts = []
    try:
        mem_system = await _get_simple_memory_system()
        if mem_system:
            # Use PostgreSQL memory system
            if LLM_PROVIDER == "ollama":
                try:
                    embed_data = _ollama_post(
                        "/api/embeddings",
                        {"model": OLLAMA_EMBED_MODEL, "prompt": body.message},
                        timeout=300
                    )
                    query_embedding = embed_data.get("embedding") or embed_data.get("embeddings", [])
                except Exception as e:
                    return {"answer": f"Embedding generation failed: {e}"}
            else:
                # OpenAI embedding
                client = _get_openai_client(llm_key)
                if client:
                    try:
                        embed_res = client.embeddings.create(
                            model=EMBED_MODEL,
                            input=body.message
                        )
                        query_embedding = embed_res.data[0].embedding
                    except Exception as e:
                        return {"answer": f"OpenAI embedding failed: {e}"}
                else:
                    query_embedding = []
            
            if query_embedding:
                # Query PostgreSQL
                try:
                    result = await mem_system.query_similar(body.message, query_embedding, k=body.top_k)
                    if result.get("success"):
                        contexts = result.get("snippets", [])
                except Exception as e:
                    # Fall back to old method
                    pass
    except Exception:
        # Fall back to old method
        pass
    
    # Fallback to old retrieval method if PostgreSQL failed
    if not contexts:
        client = _get_openai_client(llm_key) if LLM_PROVIDER != "ollama" else None
        contexts = _retrieve(client, body.message, k=max(1, min(10, body.top_k)))
    
    # Format context documents
    if contexts and hasattr(contexts[0], 'get'):
        # PostgreSQL results
        docs = "\n\n".join([f"[Doc {i+1}]\n" + (c.get("preview", "") or "") for i, c in enumerate(contexts)])
    else:
        # Old format results
        docs = "\n\n".join([f"[Doc {i+1}]\n" + (c.get("text", "") or "") for i, c in enumerate(contexts)])
    
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
async def memory_store(body: StoreBody, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    """Store user input and response in memory with embeddings"""
    try:
        # Try PostgreSQL memory system first
        mem_system = await _get_simple_memory_system()
        if mem_system:
            try:
                # Use PostgreSQL memory system
                from src.layers.memory_system import Interaction
                from datetime import datetime
                
                # Create interaction object
                interaction = Interaction(
                    user_input=body.user_input,
                    response=body.response,
                    tool_calls=[],
                    metadata={},
                    timestamp=datetime.now()
                )
                
                # Generate embedding
                if LLM_PROVIDER == "ollama":
                    try:
                        embed_data = _ollama_post(
                            "/api/embeddings",
                            {"model": OLLAMA_EMBED_MODEL, "prompt": body.user_input + "\n" + body.response},
                            timeout=300
                        )
                        embedding = embed_data.get("embedding") or embed_data.get("embeddings", [])
                    except Exception as e:
                        return {"success": False, "error": f"Embedding generation failed: {e}"}
                else:
                    # OpenAI embedding
                    client = _get_openai_client(llm_key)
                    if client:
                        try:
                            embed_res = client.embeddings.create(
                                model=EMBED_MODEL,
                                input=body.user_input + "\n" + body.response
                            )
                            embedding = embed_res.data[0].embedding
                        except Exception as e:
                            return {"success": False, "error": f"OpenAI embedding failed: {e}"}
                    else:
                        return {"success": False, "error": "No embedding provider available"}
                
                # Store in PostgreSQL
                try:
                        # Preserve original filename in content if provided via metadata
                        result = await mem_system.store_interaction(interaction, embedding)
                    if result.get("success"):
                        return {
                            "success": True,
                            "message": "Stored in PostgreSQL vector database",
                            "record_id": result.get("record_id"),
                            "type": "postgresql_pgvector"
                        }
                    else:
                        return {"success": False, "error": f"PostgreSQL storage failed: {result.get('error')}"}
                except Exception as e:
                    return {"success": False, "error": f"PostgreSQL operation failed: {e}"}
                
            except Exception as e:
                return {"success": False, "error": f"PostgreSQL memory system error: {e}"}
        
        # Fallback to JSON file storage
        mem = _load_memory()
        items = mem.get("items", [])
        
        # Generate embedding for similarity search
        if LLM_PROVIDER == "ollama":
            try:
                embed_data = _ollama_post(
                    "/api/embeddings",
                    {"model": OLLAMA_EMBED_MODEL, "prompt": body.user_input + "\n" + body.response},
                    timeout=300
                )
                embedding = embed_data.get("embedding") or embed_data.get("embeddings", [])
            except Exception as e:
                return {"success": False, "error": f"Embedding generation failed: {e}"}
        else:
            # OpenAI embedding
            client = _get_openai_client(llm_key)
            if client:
                try:
                    embed_res = client.embeddings.create(
                        model=EMBED_MODEL,
                        input=body.user_input + "\n" + body.response
                    )
                    embedding = embed_res.data[0].embedding
                except Exception as e:
                    return {"success": False, "error": f"OpenAI embedding failed: {e}"}
            else:
                embedding = []
        
        # Store in JSON file
        item = {
            "user_input": body.user_input,
            "response": body.response,
            "embedding": embedding,
            "timestamp": time.time(),
            "metadata": {}
        }
        items.append(item)
        _save_memory({"items": items})
        
        return {
            "success": True,
            "message": "Stored in JSON file (PostgreSQL not available)",
            "type": "json_file",
            "total_items": len(items)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/memory/query")
async def memory_query(query: str, k: int = 5, _: None = Depends(require_api_key), llm_key: Optional[str] = Depends(get_llm_key)) -> Dict[str, Any]:
    """Query memory for similar content using simple PostgreSQL search"""
    try:
        # Try simple PostgreSQL memory system first
        mem_system = await _get_simple_memory_system()
        if mem_system:
            try:
                results = await mem_system.query_similar(query, k)
                return {
                    "success": True,
                    "type": "postgresql_pgvector",
                    "query": query,
                    "total_found": len(results),
                    "snippets": results
                }
            except Exception as e:
                print(f"Simple PostgreSQL query failed: {e}")
        
        # Fallback to JSON file storage
        mem = _load_memory()
        items = mem.get("items", [])
        if not items:
            return {"success": True, "type": "json_file", "query": query, "snippets": [], "total_found": 0}
        
        # Simple text search in JSON items
        results = []
        for item in items:
            if query.lower() in item.get("user_input", "").lower() or query.lower() in item.get("response", "").lower():
                results.append(item)
                if len(results) >= k:
                    break
        
        return {
            "success": True,
            "type": "json_file",
            "query": query,
            "total_found": len(results),
            "snippets": results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/memory/documents")
async def get_available_documents(limit: int = Query(10, description="Number of documents to return", ge=1, le=50), _: None = Depends(require_api_key)) -> Dict[str, Any]:
    """Get available documents for preview"""
    try:
        # Try simple PostgreSQL memory system first
        mem_system = await _get_simple_memory_system()
        if mem_system:
            try:
                documents = await mem_system.get_sample_documents(limit)
                doc_count = await mem_system.get_document_count()
                return {
                    "success": True,
                    "type": "postgresql_pgvector",
                    "total_documents": doc_count,
                    "documents": documents
                }
            except Exception as e:
                print(f"Failed to get documents from PostgreSQL: {e}")
        
        # Fallback to JSON file
        mem = _load_memory()
        items = mem.get("items", [])
        return {
            "success": True,
            "type": "json_file",
            "total_documents": len(items),
            "documents": items[:limit]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/memory/document/{doc_id}")
async def get_document(doc_id: int, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    """Fetch a single document by id with full content."""
    try:
        mem_system = await _get_simple_memory_system()
        if mem_system:
            doc = await mem_system.get_document_by_id(doc_id)
            if doc:
                return {"success": True, "type": "postgresql_pgvector", "document": doc}
            return {"success": False, "error": "not_found"}
        # Fallback JSON: derive from list index
        mem = _load_memory()
        items = mem.get("items", [])
        idx = max(0, min(len(items) - 1, doc_id - 1))
        if 0 <= idx < len(items):
            it = items[idx]
            return {"success": True, "type": "json_file", "document": {"id": str(doc_id), "source": "json_file", "content": it.get("response", ""), "created_at": None}}
        return {"success": False, "error": "not_found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    return _components_snapshot()


def _components_snapshot() -> Dict[str, Any]:
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
    # Memory system status
    try:
        mem_info = _load_memory()
        if mem_info.get("type") == "postgresql_pgvector":
            # PostgreSQL memory system
            components["memory"] = {
                "type": "postgresql_pgvector",
                "database_url": mem_info.get("database_url", "not_set"),
                "embedding_dimension": mem_info.get("embedding_dimension", EMBED_DIM),
                "available": mem_info.get("available", False),
                "status": "active" if mem_info.get("available") else "error"
            }
        else:
            # JSON file fallback
            components["memory"] = {
                "type": "json_file",
                "path": str(MEMORY_PATH),
                "items": mem_info.get("items", []),
                "status": "fallback"
            }
    except Exception as e:
        components["memory"] = {"error": str(e), "status": "error"}
    
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
            
            # Try to store in PostgreSQL memory system first
            try:
                mem_system = await _get_simple_memory_system()
                if mem_system:
                    # Use PostgreSQL memory system
                    from src.layers.memory_system import Interaction
                    from datetime import datetime
                    
                    # Create interaction object
                    original_name = (uf.filename or "file")
                    interaction = Interaction(
                        user_input=f"upload:{original_name} {source or ''}",
                        response=f"[filename: {original_name}]\n{text}",
                        tool_calls=[],
                        metadata={"source": "file_upload", "filename": original_name},
                        timestamp=datetime.now()
                    )
                    
                    # Generate embedding
                    if LLM_PROVIDER == "ollama":
                        try:
                            embed_data = _ollama_post(
                                "/api/embeddings",
                                {"model": OLLAMA_EMBED_MODEL, "prompt": text},
                                timeout=300
                            )
                            embedding = embed_data.get("embedding") or embed_data.get("embeddings", [])
                        except Exception as e:
                            details.append({"file": uf.filename, "ok": False, "error": f"Embedding failed: {e}"})
                            errs += 1
                            continue
                    else:
                        # OpenAI embedding
                        client = _get_openai_client(llm_key)
                        if client:
                            try:
                                embed_res = client.embeddings.create(
                                    model=EMBED_MODEL,
                                    input=text
                                )
                                embedding = embed_res.data[0].embedding
                            except Exception as e:
                                details.append({"file": uf.filename, "ok": False, "error": f"OpenAI embedding failed: {e}"})
                                errs += 1
                                continue
                        else:
                            details.append({"file": uf.filename, "ok": False, "error": "No embedding provider available"})
                            errs += 1
                            continue
                    
                    # Store in PostgreSQL
                    try:
                        result = await mem_system.store_interaction(interaction, embedding)
                        if result.get("success"):
                            details.append({"file": uf.filename, "ok": True, "chars": len(text), "type": "postgresql_pgvector"})
                            stored += 1
                            continue
                        else:
                            details.append({"file": uf.filename, "ok": False, "error": f"PostgreSQL storage failed: {result.get('error')}"})
                            errs += 1
                            continue
                    except Exception as e:
                        details.append({"file": uf.filename, "ok": False, "error": f"PostgreSQL operation failed: {e}"})
                        errs += 1
                        continue
                        
            except Exception as e:
                # Fall back to old method
                pass
            
            # Fallback to old JSON file storage
            body = StoreBody(user_input=f"upload:{uf.filename} {source or ''}", response=text)
            await memory_store(body, None, llm_key)  # reuse in-process
            details.append({"file": uf.filename, "ok": True, "chars": len(text), "type": "json_file"})
            stored += 1
            
        except Exception as e:
            errs += 1
            details.append({"file": uf.filename, "ok": False, "error": str(e)})
    
    return IngestResponse(stored=stored, errors=errs, details=details)



