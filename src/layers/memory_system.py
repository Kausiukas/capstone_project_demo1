from __future__ import annotations

import os
import hashlib
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from src.modules.module_2_support.postgresql_vector_agent import PostgreSQLVectorAgent

logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    user_input: str
    response: str
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime


class ShortTermWindow:
    """Simple short-term memory window for recent interactions."""
    
    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars
        self.turns: List[Dict[str, str]] = []
        self._current_chars = 0
    
    def add_turn(self, user_input: str, response: str):
        turn = {"user": user_input, "assistant": response}
        turn_chars = len(user_input) + len(response)
        
        # Add new turn
        self.turns.append(turn)
        self._current_chars += turn_chars
        
        # Remove old turns if we exceed max_chars
        while self._current_chars > self.max_chars and len(self.turns) > 1:
            old_turn = self.turns.pop(0)
            self._current_chars -= (len(old_turn["user"]) + len(old_turn["assistant"]))
    
    def render(self) -> Dict[str, Any]:
        if not self.turns:
            return {"summary": "", "compressed": False}
        
        # Simple summary of recent turns
        summary_parts = []
        for turn in self.turns[-3:]:  # Last 3 turns
            summary_parts.append(f"User: {turn['user'][:100]}...")
            summary_parts.append(f"Assistant: {turn['assistant'][:100]}...")
        
        summary = "\n".join(summary_parts)
        compressed = self._current_chars > self.max_chars * 0.8
        
        return {
            "summary": summary,
            "compressed": compressed,
            "turns_count": len(self.turns),
            "total_chars": self._current_chars
        }


class MemorySystem:
    """Enhanced memory layer: store and retrieve interactions using pgvector with improved connection handling."""

    def __init__(self, connection_string: Optional[str] = None, embedding_dimension: int = 1536,
                 window_max_chars: int = 2000):
        self.connection_string = connection_string
        self.embedding_dimension = embedding_dimension
        self.window_max_chars = window_max_chars
        
        # Initialize components
        self.agent = PostgreSQLVectorAgent(
            connection_string=connection_string or os.getenv("DATABASE_URL"),
            embedding_dimension=embedding_dimension
        )
        self.window = ShortTermWindow(max_chars=window_max_chars)
        
        # Connection state
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._connection_retry_count = 0
        self._max_retries = 3
        
        # Health monitoring
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the memory system with proper error handling and retries."""
        if self._initialized:
            return {"success": True, "message": "Already initialized"}
        
        async with self._initialization_lock:
            if self._initialized:  # Double-check pattern
                return {"success": True, "message": "Already initialized"}
            
            try:
                logger.info("Initializing MemorySystem...")
                
                # Initialize PostgreSQL agent
                result = await self.agent.initialize()
                if not result.get("success"):
                    error_msg = f"PostgreSQL agent initialization failed: {result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
                
                self._initialized = True
                self._connection_retry_count = 0
                logger.info("MemorySystem initialized successfully")
                
                return {
                    "success": True,
                    "message": "MemorySystem initialized successfully",
                    "embedding_dimension": self.embedding_dimension
                }
                
            except Exception as e:
                error_msg = f"MemorySystem initialization failed: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
    
    async def _ensure_initialized(self) -> bool:
        """Ensure the system is initialized, with retry logic."""
        if self._initialized:
            return True
        
        # Try to initialize
        result = await self.initialize()
        if result.get("success"):
            return True
        
        # If initialization failed, check if we should retry
        if self._connection_retry_count < self._max_retries:
            self._connection_retry_count += 1
            logger.warning(f"Initialization failed, retry {self._connection_retry_count}/{self._max_retries}")
            
            # Wait before retry
            await asyncio.sleep(1 * self._connection_retry_count)
            
            # Try again
            result = await self.initialize()
            if result.get("success"):
                return True
        
        logger.error("Failed to initialize MemorySystem after all retries")
        return False
    
    async def _health_check(self) -> bool:
        """Perform health check on the PostgreSQL connection."""
        try:
            if not self._initialized:
                return False
            
            current_time = datetime.now().timestamp()
            if current_time - self._last_health_check < self._health_check_interval:
                return True  # Skip if too recent
            
            # Perform actual health check
            health_result = await self.agent.health_check()
            is_healthy = health_result.get("success", False)
            
            self._last_health_check = current_time
            
            if not is_healthy:
                logger.warning("PostgreSQL health check failed, marking as unhealthy")
                self._initialized = False
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._initialized = False
            return False
    
    async def store_interaction(self, interaction: Interaction, embedding: List[float]) -> Dict[str, Any]:
        """Store interaction with improved error handling and connection management."""
        try:
            # Ensure system is initialized
            if not await self._ensure_initialized():
                return {
                    "success": False,
                    "error": "Failed to initialize memory system"
                }
            
            # Validate embedding
            if not embedding or len(embedding) != self.embedding_dimension:
                return {
                    "success": False,
                    "error": f"Invalid embedding dimension: got {len(embedding) if embedding else 0}, expected {self.embedding_dimension}"
                }
            
            # Prepare content and metadata
            content = interaction.user_input + "\n---\n" + interaction.response
            metadata = {
                "tool_calls": interaction.tool_calls,
                "timestamp": interaction.timestamp.isoformat(),
                **(interaction.metadata or {}),
                "feedback_up": int((interaction.metadata or {}).get("feedback_up", 0)),
                "feedback_down": int((interaction.metadata or {}).get("feedback_down", 0)),
            }
            
            # Store in PostgreSQL
            result = await self.agent.store_vector(
                content=content,
                embedding=embedding,
                metadata=metadata,
                tags=["interaction"],
                source="conversation"
            )
            
            if result.get("success"):
                # Add to short-term window
                self.window.add_turn(interaction.user_input, interaction.response)
                
                logger.info(f"Successfully stored interaction with ID: {result.get('record_id')}")
                return result
            else:
                logger.error(f"Failed to store interaction: {result.get('error')}")
                return result
                
        except Exception as e:
            error_msg = f"Error storing interaction: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    async def query_context(self, query_text: str, query_embedding: List[float], k: int = 5,
                            include_window: bool = True) -> Dict[str, Any]:
        """Query context with improved error handling and connection management."""
        try:
            # Ensure system is initialized
            if not await self._ensure_initialized():
                return {
                    "success": False,
                    "error": "Failed to initialize memory system"
                }
            
            # Validate embedding
            if not query_embedding or len(query_embedding) != self.embedding_dimension:
                return {
                    "success": False,
                    "error": f"Invalid query embedding dimension: got {len(query_embedding) if query_embedding else 0}, expected {self.embedding_dimension}"
                }
            
            # Check connection health
            if not await self._health_check():
                return {
                    "success": False,
                    "error": "PostgreSQL connection is unhealthy"
                }
            
            # Query PostgreSQL
            result = await self.agent.search_similar(query_embedding, limit=k * 3)
            if not result.get("success"):
                logger.error(f"PostgreSQL search failed: {result.get('error')}")
                return result
            
            # Process and rank results
            ranked = await self._rank_results(result.get("results", []), query_embedding)
            top_results = ranked[:k]
            
            # Build response
            snippets = []
            
            # Add short-term window if requested
            if include_window:
                window_info = self.window.render()
                if window_info.get("summary"):
                    snippets.append({
                        "id": "short_term_window",
                        "similarity": None,
                        "adjusted_score": None,
                        "preview": window_info["summary"][:500],
                        "source": "short_term",
                        "tags": ["window", "autosummary"] + (["compressed"] if window_info.get("compressed") else []),
                    })
            
            # Add ranked results
            for score, result_item, text in top_results:
                snippets.append({
                    "id": result_item.record.id,
                    "similarity": result_item.similarity,
                    "adjusted_score": score,
                    "preview": text[:500],
                    "source": result_item.record.source,
                    "tags": result_item.record.tags,
                })
            
            logger.info(f"Successfully retrieved {len(snippets)} context snippets")
            return {"success": True, "snippets": snippets}
            
        except Exception as e:
            error_msg = f"Error querying context: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    async def _rank_results(self, results: List, query_embedding: List[float]) -> List[tuple]:
        """Rank results with feedback adjustment."""
        try:
            ranked = []
            for result_item in results:
                text = result_item.record.content
                meta = result_item.record.metadata or {}
                
                # Calculate feedback-adjusted score
                up = int(meta.get("feedback_up", 0))
                down = int(meta.get("feedback_down", 0))
                adjusted = float(result_item.similarity) * (1.0 + 0.15 * up - 0.25 * down)
                
                ranked.append((adjusted, result_item, text))
            
            # Sort by adjusted score
            ranked.sort(key=lambda x: x[0], reverse=True)
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            # Return original results if ranking fails
            return [(r.similarity, r, r.record.content) for r in results]
    
    async def apply_feedback(self, record_id: str, feedback: str) -> Dict[str, Any]:
        """Apply feedback to a memory record with improved error handling."""
        try:
            # Ensure system is initialized
            if not await self._ensure_initialized():
                return {
                    "success": False,
                    "error": "Failed to initialize memory system"
                }
            
            # Validate feedback
            if feedback not in ["up", "down"]:
                return {
                    "success": False,
                    "error": "Invalid feedback value, use 'up' or 'down'"
                }
            
            # Get current record
            rec = await self.agent.get_vector_by_id(record_id)
            if not rec.get("success"):
                return rec
            
            record = rec["record"]
            meta = dict(record.metadata or {})
            
            # Update feedback counters
            if feedback == "up":
                meta["feedback_up"] = int(meta.get("feedback_up", 0)) + 1
            else:  # feedback == "down"
                meta["feedback_down"] = int(meta.get("feedback_down", 0)) + 1
            
            # Update record
            upd = await self.agent.update_vector(record_id=record_id, metadata=meta)
            if not upd.get("success"):
                return upd
            
            logger.info(f"Successfully applied {feedback} feedback to record {record_id}")
            return {
                "success": True,
                "record_id": record_id,
                "metadata": meta,
                "feedback_applied": feedback
            }
            
        except Exception as e:
            error_msg = f"Error applying feedback: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    async def cleanup(self):
        """Cleanup resources and close connections."""
        try:
            if hasattr(self.agent, 'cleanup'):
                await self.agent.cleanup()
            self._initialized = False
            logger.info("MemorySystem cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        try:
            status = {
                "initialized": self._initialized,
                "connection_retry_count": self._connection_retry_count,
                "embedding_dimension": self.embedding_dimension,
                "window_info": self.window.render(),
                "last_health_check": self._last_health_check
            }
            
            if self._initialized:
                # Get PostgreSQL status
                pg_status = await self.agent.health_check()
                status["postgresql"] = pg_status
                
                # Check connection health
                is_healthy = await self._health_check()
                status["connection_healthy"] = is_healthy
            
            return {
                "success": True,
                "status": status
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting status: {str(e)}"
            }


