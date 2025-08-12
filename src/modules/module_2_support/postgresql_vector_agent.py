"""
PostgreSQL Vector Agent - Enhanced vector database operations and memory management
"""

import asyncio
import asyncpg
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import os
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class VectorRecord:
    """Vector record structure"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    source: str

@dataclass
class SearchResult:
    """Search result structure"""
    record: VectorRecord
    similarity: float
    rank: int

class PostgreSQLVectorAgent:
    """
    Enhanced PostgreSQL vector agent with advanced capabilities
    """
    
    def __init__(self, connection_string: str = None, embedding_dimension: Optional[int] = None):
        self.connection_string = connection_string
        self.pool = None
        # Allow override via argument or env var for demos/tests
        self.embedding_dimension = embedding_dimension or int(os.getenv("MEMORY_VECTOR_DIM", "1536"))
        self.table_name = "vector_store"
        self.index_name = "vector_index"
        self.initialized = False
        
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the PostgreSQL vector agent with improved connection management
        """
        try:
            if not self.connection_string:
                return {
                    "success": False,
                    "error": "Connection string not provided"
                }
            
            # Optional SSL requirement
            ssl_mode = os.getenv("PG_SSLMODE") or os.getenv("DB_SSLMODE")
            ssl_required = True if (ssl_mode and ssl_mode.lower() in {"require", "verify-full", "verify-ca"}) else False

            async def _setup_connection(conn: asyncpg.Connection):
                try:
                    await conn.execute("SET statement_timeout = 5000")
                    await conn.execute("SET idle_in_transaction_session_timeout = 10000")
                except Exception:
                    pass

            # Create connection pool with optimized settings
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,  # Increased from 1 for better availability
                max_size=8,  # Increased from 4 for better concurrency
                command_timeout=15,  # Increased from 10 for better reliability
                timeout=20,  # Connection timeout
                init=_setup_connection,
                ssl=ssl_required or None,
                server_settings={
                    'application_name': 'langflow_vector_agent',
                    'jit': 'off',  # Disable JIT for better connection stability
                }
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            # Initialize database schema
            await self._create_schema()
            
            self.initialized = True
            
            logger.info("PostgreSQL Vector Agent initialized successfully")
            return {
                "success": True,
                "message": "PostgreSQL Vector Agent initialized successfully",
                "table_name": self.table_name,
                "index_name": self.index_name,
                "pool_size": self.pool.get_size()
            }
            
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL Vector Agent: {str(e)}")
            # Cleanup on failure
            if hasattr(self, 'pool') and self.pool:
                await self.pool.close()
                self.pool = None
            return {
                "success": False,
                "error": str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the PostgreSQL vector agent
        """
        try:
            if not self.initialized or not self.pool:
                return {
                    "success": False,
                    "error": "Agent not initialized or pool not available"
                }
            
            # Test connection pool health
            async with self.pool.acquire() as conn:
                # Test basic connectivity
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    return {
                        "success": False,
                        "error": "Basic connectivity test failed"
                    }
                
                # Test table existence
                table_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                    self.table_name
                )
                
                if not table_exists:
                    return {
                        "success": False,
                        "error": f"Table {self.table_name} does not exist"
                    }
                
                # Get pool statistics
                pool_size = self.pool.get_size()
                # asyncpg.Pool doesn't have get_free_size, use available methods
                return {
                    "success": True,
                    "message": "PostgreSQL Vector Agent is healthy",
                    "pool_size": pool_size,
                    "table_name": self.table_name,
                    "index_name": self.index_name
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources and close connection pool"""
        try:
            if hasattr(self, 'pool') and self.pool:
                await self.pool.close()
                self.pool = None
                self.initialized = False
                logger.info("PostgreSQL Vector Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _get_connection(self):
        """Get a connection from the pool with error handling"""
        if not self.initialized or not self.pool:
            raise RuntimeError("PostgreSQL Vector Agent not initialized")
        
        try:
            return await self.pool.acquire()
        except Exception as e:
            logger.error(f"Failed to acquire connection: {e}")
            raise
    
    async def _create_schema(self):
        """Create database schema for vector operations"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Ensure UUID generation is available for id default
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

            # If the base table exists with a different dimension, switch to a dim-suffixed table
            try:
                fmt = await conn.fetchval(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = $1 AND a.attname = 'embedding'
                    """,
                    self.table_name,
                )
                if fmt and fmt.startswith("vector("):
                    try:
                        existing_dim = int(fmt.split("(")[1].split(")")[0])
                    except Exception:
                        existing_dim = None
                    if existing_dim and existing_dim != self.embedding_dimension:
                        # Switch to a dimension-specific table
                        self.table_name = f"vector_store_{self.embedding_dimension}"
            except Exception:
                pass

            # Derive index name from table name to avoid collisions
            self.index_name = f"{self.table_name}_ivfflat_idx"
            
            # Create vector store table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    embedding vector({self.embedding_dimension}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    tags TEXT[] DEFAULT '{{}}',
                    source VARCHAR(255),
                    content_hash VARCHAR(64) UNIQUE
                )
            """)
            
            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.index_name}
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tags 
                ON {self.table_name} USING GIN (tags)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source 
                ON {self.table_name} (source)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at 
                ON {self.table_name} (created_at)
            """)
    
    def _to_vector_text(self, embedding: List[float]) -> str:
        """Format embedding as pgvector text literal like "[0.1, 0.2, ...]".
        Ensures pure Python floats and consistent formatting.
        """
        values = []
        for v in embedding:
            try:
                fv = float(v)
            except Exception as e:
                raise ValueError(f"Embedding contains non-numeric value: {v!r}") from e
            values.append(str(fv))
        return "[" + ", ".join(values) + "]"

    async def store_vector(self, content: str, embedding: List[float], 
                          metadata: Dict[str, Any] = None, tags: List[str] = None,
                          source: str = None) -> Dict[str, Any]:
        """
        Store a vector in the database
        
        Args:
            content: Text content
            embedding: Vector embedding
            metadata: Additional metadata
            tags: Tags for categorization
            source: Source of the content
            
        Returns:
            Dictionary containing storage result
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            # Generate content hash
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if content already exists
            existing = await self._get_by_content_hash(content_hash)
            if existing:
                return {
                    "success": True,
                    "message": "Content already exists",
                    "record_id": existing["id"],
                    "duplicate": True
                }
            
            # Validate dimension
            if len(embedding) != self.embedding_dimension:
                return {
                    "success": False,
                    "error": (
                        f"Embedding dimension mismatch: got {len(embedding)}, expected {self.embedding_dimension}"
                    )
                }

            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(f"""
                    INSERT INTO {self.table_name} 
                    (content, metadata, embedding, tags, source, content_hash)
                    VALUES ($1, $2, $3::vector, $4, $5, $6)
                    RETURNING id, created_at, updated_at
                """, content, json.dumps(metadata or {}),
                     self._to_vector_text(embedding),
                     tags or [], source, content_hash)
                
                return {
                    "success": True,
                    "record_id": str(record['id']),
                    "created_at": record['created_at'],
                    "updated_at": record['updated_at']
                }
                
        except Exception as e:
            logger.error(f"Error storing vector: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_similar(self, query_embedding: List[float], 
                           limit: int = 10, threshold: float = 0.7,
                           tags: List[str] = None, source: str = None) -> Dict[str, Any]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            threshold: Similarity threshold
            tags: Filter by tags
            source: Filter by source
            
        Returns:
            Dictionary containing search results
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            # Validate dimension
            if len(query_embedding) != self.embedding_dimension:
                return {
                    "success": False,
                    "error": (
                        f"Query embedding dimension mismatch: got {len(query_embedding)}, expected {self.embedding_dimension}"
                    )
                }

            # Build query
            query = f"""
                SELECT id, content, metadata, embedding, created_at, updated_at, tags, source,
                       1 - (embedding <=> $1::vector) as similarity
                FROM {self.table_name}
                WHERE 1 - (embedding <=> $1::vector) > $2
            """
            params = [self._to_vector_text(query_embedding), threshold]
            param_count = 2
            
            if tags:
                query += f" AND tags && ${param_count + 1}"
                params.append(tags)
                param_count += 1
            
            if source:
                query += f" AND source = ${param_count + 1}"
                params.append(source)
                param_count += 1
            
            query += f" ORDER BY embedding <=> $1::vector LIMIT ${param_count + 1}"
            params.append(limit)
            
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, *params)
                
                results = []
                for i, record in enumerate(records):
                    metadata_value = record['metadata']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_value = json.loads(metadata_value)
                        except Exception:
                            metadata_value = {}
                    elif metadata_value is None:
                        metadata_value = {}

                    vector_record = VectorRecord(
                        id=str(record['id']),
                        content=record['content'],
                        metadata=metadata_value,
                        embedding=record['embedding'],
                        created_at=record['created_at'],
                        updated_at=record['updated_at'],
                        tags=record['tags'] or [],
                        source=record['source']
                    )
                    
                    search_result = SearchResult(
                        record=vector_record,
                        similarity=float(record['similarity']),
                        rank=i + 1
                    )
                    
                    results.append(search_result)
                
                return {
                    "success": True,
                    "results": results,
                    "total_found": len(results),
                    "query_threshold": threshold
                }
                
        except Exception as e:
            logger.error(f"Error searching similar vectors: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_vector_by_id(self, record_id: str) -> Dict[str, Any]:
        """
        Get vector record by ID
        
        Args:
            record_id: Record ID
            
        Returns:
            Dictionary containing vector record
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(f"""
                    SELECT id, content, metadata, embedding, created_at, updated_at, tags, source
                    FROM {self.table_name}
                    WHERE id = $1
                """, record_id)
                
                if not record:
                    return {
                        "success": False,
                        "error": f"Record with ID {record_id} not found"
                    }
                
                vector_record = VectorRecord(
                    id=str(record['id']),
                    content=record['content'],
                    metadata=json.loads(record['metadata']) if record['metadata'] else {},
                    embedding=record['embedding'],
                    created_at=record['created_at'],
                    updated_at=record['updated_at'],
                    tags=record['tags'] or [],
                    source=record['source']
                )
                
                return {
                    "success": True,
                    "record": vector_record
                }
                
        except Exception as e:
            logger.error(f"Error getting vector by ID: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_vector(self, record_id: str, content: str = None,
                           metadata: Dict[str, Any] = None, tags: List[str] = None) -> Dict[str, Any]:
        """
        Update a vector record
        
        Args:
            record_id: Record ID
            content: New content
            metadata: New metadata
            tags: New tags
            
        Returns:
            Dictionary containing update result
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            # Build update query
            updates = []
            params = []
            param_count = 0
            
            if content is not None:
                updates.append(f"content = ${param_count + 1}")
                params.append(content)
                param_count += 1
            
            if metadata is not None:
                updates.append(f"metadata = ${param_count + 1}")
                params.append(json.dumps(metadata))
                param_count += 1
            
            if tags is not None:
                updates.append(f"tags = ${param_count + 1}")
                params.append(tags)
                param_count += 1
            
            if not updates:
                return {
                    "success": False,
                    "error": "No fields to update"
                }
            
            updates.append(f"updated_at = NOW()")
            params.append(record_id)
            
            query = f"""
                UPDATE {self.table_name}
                SET {', '.join(updates)}
                WHERE id = ${param_count + 1}
                RETURNING id, updated_at
            """
            
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(query, *params)
                
                if not record:
                    return {
                        "success": False,
                        "error": f"Record with ID {record_id} not found"
                    }
                
                return {
                    "success": True,
                    "record_id": str(record['id']),
                    "updated_at": record['updated_at']
                }
                
        except Exception as e:
            logger.error(f"Error updating vector: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_vector(self, record_id: str) -> Dict[str, Any]:
        """
        Delete a vector record
        
        Args:
            record_id: Record ID
            
        Returns:
            Dictionary containing deletion result
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            async with self.pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE id = $1
                """, record_id)
                
                if result == "DELETE 0":
                    return {
                        "success": False,
                        "error": f"Record with ID {record_id} not found"
                    }
                
                return {
                    "success": True,
                    "message": f"Record {record_id} deleted successfully"
                }
                
        except Exception as e:
            logger.error(f"Error deleting vector: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary containing statistics
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            async with self.pool.acquire() as conn:
                # Total records
                total_records = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM {self.table_name}
                """)
                
                # Records by source
                sources = await conn.fetch(f"""
                    SELECT source, COUNT(*) as count
                    FROM {self.table_name}
                    WHERE source IS NOT NULL
                    GROUP BY source
                    ORDER BY count DESC
                """)
                
                # Tags distribution
                tags = await conn.fetch(f"""
                    SELECT unnest(tags) as tag, COUNT(*) as count
                    FROM {self.table_name}
                    WHERE tags IS NOT NULL AND array_length(tags, 1) > 0
                    GROUP BY tag
                    ORDER BY count DESC
                    LIMIT 20
                """)
                
                # Recent activity
                recent_activity = await conn.fetch(f"""
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM {self.table_name}
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """)
                
                return {
                    "success": True,
                    "statistics": {
                        "total_records": total_records,
                        "sources": [{"source": r["source"], "count": r["count"]} for r in sources],
                        "top_tags": [{"tag": r["tag"], "count": r["count"]} for r in tags],
                        "recent_activity": [{"date": str(r["date"]), "count": r["count"]} for r in recent_activity]
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get record by content hash"""
        try:
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE content_hash = $1
                """, content_hash)
                
                if record:
                    return {
                        "id": str(record['id']),
                        "content": record['content'],
                        "metadata": json.loads(record['metadata']) if record['metadata'] else {},
                        "created_at": record['created_at'],
                        "updated_at": record['updated_at']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting by content hash: {str(e)}")
            return None
    
    async def cleanup_old_records(self, days: int = 90) -> Dict[str, Any]:
        """
        Clean up old records
        
        Args:
            days: Number of days to keep records
            
        Returns:
            Dictionary containing cleanup result
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "Agent not initialized"
                }
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            async with self.pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE created_at < $1
                """, cutoff_date)
                
                deleted_count = int(result.split()[1])
                
                return {
                    "success": True,
                    "deleted_count": deleted_count,
                    "cutoff_date": cutoff_date.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error cleaning up old records: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close() 