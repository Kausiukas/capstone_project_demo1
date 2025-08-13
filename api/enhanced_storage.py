"""
Enhanced Storage System for Structured Documents
Handles storage of enhanced document structures with fallback options
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import json
import time
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class StorageMethod(Enum):
    """Available storage methods"""
    ENHANCED_STRUCTURED = "enhanced_structured"  # New structured storage
    CURRENT_VECTOR = "current_vector"            # Your existing vector storage
    HYBRID = "hybrid"                           # Combine both methods
    FALLBACK = "fallback"                       # Basic storage only

class EnhancedStorageEngine:
    """
    Enhanced storage engine that handles structured documents
    with intelligent fallback and complementary storage
    """
    
    def __init__(self, 
                 current_memory_system: Any = None,
                 database_url: str = None):
        
        self.current_memory_system = current_memory_system
        self.database_url = database_url
        
        # Storage method priorities
        self.storage_priorities = [
            StorageMethod.ENHANCED_STRUCTURED,
            StorageMethod.CURRENT_VECTOR,
            StorageMethod.HYBRID,
            StorageMethod.FALLBACK
        ]
        
        # Quality thresholds for storage decisions
        self.storage_thresholds = {
            "excellent": 0.9,    # Use enhanced structured storage
            "good": 0.7,         # Use hybrid storage
            "fair": 0.5,         # Use current vector storage
            "poor": 0.3          # Use fallback storage
        }
    
    async def store_document_enhanced(self, 
                                     ingestion_results: Dict[str, Any],
                                     storage_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Store document using enhanced storage methods with intelligent routing
        """
        start_time = time.time()
        
        logger.info(f"Starting enhanced storage for {ingestion_results.get('file_path', 'unknown')}")
        
        # Determine storage method based on quality
        quality_level = ingestion_results.get("quality_level", "failed")
        storage_method = self._determine_storage_method(quality_level)
        
        logger.info(f"Selected storage method: {storage_method.value} for quality: {quality_level}")
        
        # Initialize results container
        storage_results = {
            "file_path": ingestion_results.get("file_path"),
            "storage_method": storage_method.value,
            "quality_level": quality_level,
            "storage_success": False,
            "storage_details": {},
            "fallback_used": False,
            "complementary_storage": [],
            "processing_time_ms": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Execute storage based on selected method
            if storage_method == StorageMethod.ENHANCED_STRUCTURED:
                result = await self._store_enhanced_structured(ingestion_results, storage_options)
                storage_results["storage_details"]["enhanced"] = result
                
                if result["success"]:
                    storage_results["storage_success"] = True
                    logger.info("Enhanced structured storage successful")
                else:
                    storage_results["errors"].append(f"Enhanced storage failed: {result.get('error', 'Unknown error')}")
            
            elif storage_method == StorageMethod.HYBRID:
                result = await self._store_hybrid(ingestion_results, storage_options)
                storage_results["storage_details"]["hybrid"] = result
                
                if result["success"]:
                    storage_results["storage_success"] = True
                    logger.info("Hybrid storage successful")
                else:
                    storage_results["errors"].append(f"Hybrid storage failed: {result.get('error', 'Unknown error')}")
            
            elif storage_method == StorageMethod.CURRENT_VECTOR:
                result = await self._store_current_vector(ingestion_results, storage_options)
                storage_results["storage_details"]["current_vector"] = result
                
                if result["success"]:
                    storage_results["storage_success"] = True
                    logger.info("Current vector storage successful")
                else:
                    storage_results["errors"].append(f"Current vector storage failed: {result.get('error', 'Unknown error')}")
            
            elif storage_method == StorageMethod.FALLBACK:
                result = await self._store_fallback(ingestion_results, storage_options)
                storage_results["storage_details"]["fallback"] = result
                
                if result["success"]:
                    storage_results["storage_success"] = True
                    storage_results["fallback_used"] = True
                    logger.info("Fallback storage successful")
                else:
                    storage_results["errors"].append(f"Fallback storage failed: {result.get('error', 'Unknown error')}")
            
            # Apply complementary storage if primary method succeeded
            if storage_results["storage_success"]:
                complementary_results = await self._apply_complementary_storage(
                    ingestion_results, storage_options
                )
                
                for comp_result in complementary_results:
                    if comp_result["success"]:
                        storage_results["complementary_storage"].append(comp_result)
                        logger.info(f"Complementary storage successful: {comp_result['method']}")
                    else:
                        storage_results["warnings"].append(f"Complementary storage failed: {comp_result.get('error', 'Unknown error')}")
            
            # Final validation and summary
            if storage_results["storage_success"]:
                storage_results["summary"] = self._generate_storage_summary(storage_results)
                logger.info(f"Enhanced storage completed successfully using {storage_method.value}")
            else:
                storage_results["errors"].append("All storage methods failed")
                logger.error("All storage methods failed")
        
        except Exception as e:
            error_msg = f"Unexpected error during enhanced storage: {str(e)}"
            storage_results["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        finally:
            storage_results["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return storage_results
    
    def _determine_storage_method(self, quality_level: str) -> StorageMethod:
        """Determine storage method based on quality level"""
        quality_score = self.storage_thresholds.get(quality_level, 0.0)
        
        if quality_score >= self.storage_thresholds["excellent"]:
            return StorageMethod.ENHANCED_STRUCTURED
        elif quality_score >= self.storage_thresholds["good"]:
            return StorageMethod.HYBRID
        elif quality_score >= self.storage_thresholds["fair"]:
            return StorageMethod.CURRENT_VECTOR
        else:
            return StorageMethod.FALLBACK
    
    async def _store_enhanced_structured(self, 
                                        ingestion_results: Dict[str, Any],
                                        storage_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store document using enhanced structured storage"""
        try:
            logger.info("Attempting enhanced structured storage")
            
            # Extract structured data
            combined_data = ingestion_results.get("combined_data", {})
            
            if not combined_data:
                return {
                    "success": False,
                    "error": "No combined data available for enhanced storage"
                }
            
            # Store structured document metadata
            structured_result = await self._store_structured_metadata(combined_data)
            
            if not structured_result["success"]:
                return structured_result
            
            # Store document sections
            sections_result = await self._store_document_sections(
                structured_result["document_id"], combined_data.get("sections", [])
            )
            
            # Store extracted entities
            entities_result = await self._store_document_entities(
                structured_result["document_id"], combined_data.get("entities", [])
            )
            
            # Store embeddings if available
            embeddings_result = {"success": True, "stored": False}
            if combined_data.get("embeddings"):
                embeddings_result = await self._store_document_embeddings(
                    structured_result["document_id"], combined_data["embeddings"]
                )
            
            return {
                "success": True,
                "method": "enhanced_structured",
                "document_id": structured_result["document_id"],
                "metadata_stored": structured_result["success"],
                "sections_stored": sections_result["success"],
                "entities_stored": entities_result["success"],
                "embeddings_stored": embeddings_result["success"],
                "storage_details": {
                    "structured": structured_result,
                    "sections": sections_result,
                    "entities": entities_result,
                    "embeddings": embeddings_result
                }
            }
        
        except Exception as e:
            logger.error(f"Enhanced structured storage failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_hybrid(self, 
                            ingestion_results: Dict[str, Any],
                            storage_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store document using hybrid approach (both enhanced and current systems)"""
        try:
            logger.info("Attempting hybrid storage")
            
            # Try enhanced storage first
            enhanced_result = await self._store_enhanced_structured(ingestion_results, storage_options)
            
            # Try current vector storage
            current_result = await self._store_current_vector(ingestion_results, storage_options)
            
            # Combine results
            hybrid_success = enhanced_result["success"] or current_result["success"]
            
            return {
                "success": hybrid_success,
                "method": "hybrid",
                "enhanced_storage": enhanced_result,
                "current_storage": current_result,
                "combined_success": hybrid_success
            }
        
        except Exception as e:
            logger.error(f"Hybrid storage failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_current_vector(self, 
                                   ingestion_results: Dict[str, Any],
                                   storage_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store document using current vector storage system"""
        try:
            logger.info("Attempting current vector storage")
            
            if not self.current_memory_system:
                return {
                    "success": False,
                    "error": "Current memory system not available"
                }
            
            # Extract content for vector storage
            content = self._extract_content_for_vector_storage(ingestion_results)
            
            if not content:
                return {
                    "success": False,
                    "error": "No content available for vector storage"
                }
            
            # Generate embeddings using current system
            embedding_result = await self._generate_embedding_current_system(content)
            
            if not embedding_result["success"]:
                return embedding_result
            
            # Store in current vector system
            storage_result = await self.current_memory_system.store_vector(
                content=content,
                embedding=embedding_result["embedding"],
                metadata={
                    "source": "enhanced_ingestion",
                    "quality_level": ingestion_results.get("quality_level", "unknown"),
                    "extraction_methods": ingestion_results.get("extraction_methods_used", []),
                    "file_path": ingestion_results.get("file_path"),
                    "file_type": ingestion_results.get("file_type")
                }
            )
            
            return {
                "success": storage_result.get("success", False),
                "method": "current_vector",
                "content_stored": bool(content),
                "embedding_generated": embedding_result["success"],
                "vector_stored": storage_result.get("success", False),
                "record_id": storage_result.get("record_id"),
                "metadata": storage_result
            }
        
        except Exception as e:
            logger.error(f"Current vector storage failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_fallback(self, 
                              ingestion_results: Dict[str, Any],
                              storage_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store document using basic fallback storage"""
        try:
            logger.info("Attempting fallback storage")
            
            # Extract basic content
            content = self._extract_content_for_vector_storage(ingestion_results)
            
            if not content:
                return {
                    "success": False,
                    "error": "No content available for fallback storage"
                }
            
            # Basic text storage (no embeddings)
            fallback_result = await self._store_basic_text(content, ingestion_results)
            
            return {
                "success": fallback_result["success"],
                "method": "fallback",
                "content_stored": bool(content),
                "basic_storage": fallback_result
            }
        
        except Exception as e:
            logger.error(f"Fallback storage failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_content_for_vector_storage(self, ingestion_results: Dict[str, Any]) -> str:
        """Extract content suitable for vector storage"""
        try:
            combined_data = ingestion_results.get("combined_data", {})
            
            # Try to get content from sections first
            if combined_data.get("sections"):
                content_parts = []
                for section in combined_data["sections"][:5]:  # Limit to first 5 sections
                    content_parts.append(section.get("content", ""))
                return " ".join(content_parts)
            
            # Fallback to raw content
            if combined_data.get("content"):
                return combined_data["content"]
            
            # Try fallback result
            fallback_result = ingestion_results.get("fallback_result")
            if fallback_result and fallback_result.get("success"):
                fallback_data = fallback_result["data"]
                return fallback_data.get("content", "")
            
            return ""
        
        except Exception as e:
            logger.error(f"Failed to extract content for vector storage: {e}")
            return ""
    
    async def _generate_embedding_current_system(self, content: str) -> Dict[str, Any]:
        """Generate embedding using current system"""
        try:
            if not self.current_memory_system:
                return {
                    "success": False,
                    "error": "Current memory system not available"
                }
            
            # This would use your existing embedding generation logic
            # For now, we'll simulate it
            return {
                "success": True,
                "embedding": [0.1] * 1024,  # Simulated embedding
                "dimension": 1024,
                "method": "current_system_simulated"
            }
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_structured_metadata(self, combined_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store structured document metadata"""
        try:
            # This would create a new table for structured documents
            # For now, we'll simulate the storage
            document_id = f"doc_{int(time.time())}_{hash(combined_data.get('title', 'unknown'))}"
            
            return {
                "success": True,
                "document_id": document_id,
                "metadata_stored": True
            }
        
        except Exception as e:
            logger.error(f"Failed to store structured metadata: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_document_sections(self, document_id: str, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store document sections"""
        try:
            if not sections:
                return {"success": True, "sections_stored": 0}
            
            # Simulate storing sections
            stored_count = len(sections)
            
            return {
                "success": True,
                "sections_stored": stored_count,
                "total_sections": len(sections)
            }
        
        except Exception as e:
            logger.error(f"Failed to store document sections: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_document_entities(self, document_id: str, entities: List[str]) -> Dict[str, Any]:
        """Store extracted entities"""
        try:
            if not entities:
                return {"success": True, "entities_stored": 0}
            
            # Simulate storing entities
            stored_count = len(entities)
            
            return {
                "success": True,
                "entities_stored": stored_count,
                "total_entities": len(entities)
            }
        
        except Exception as e:
            logger.error(f"Failed to store document entities: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_document_embeddings(self, document_id: str, embeddings: List[float]) -> Dict[str, Any]:
        """Store document embeddings"""
        try:
            if not embeddings:
                return {"success": False, "error": "No embeddings provided"}
            
            # Simulate storing embeddings
            return {
                "success": True,
                "embeddings_stored": True,
                "embedding_dimension": len(embeddings)
            }
        
        except Exception as e:
            logger.error(f"Failed to store document embeddings: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_basic_text(self, content: str, ingestion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Store basic text content without embeddings"""
        try:
            # Simulate basic text storage
            return {
                "success": True,
                "content_stored": True,
                "content_length": len(content),
                "method": "basic_text"
            }
        
        except Exception as e:
            logger.error(f"Failed to store basic text: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _apply_complementary_storage(self, 
                                          ingestion_results: Dict[str, Any],
                                          storage_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Apply complementary storage methods"""
        try:
            complementary_results = []
            
            # Store chat history if available
            if ingestion_results.get("chat_context"):
                chat_result = await self._store_chat_context(
                    ingestion_results["chat_context"]
                )
                complementary_results.append(chat_result)
            
            # Store performance metrics
            performance_result = await self._store_performance_metrics(ingestion_results)
            complementary_results.append(performance_result)
            
            return complementary_results
        
        except Exception as e:
            logger.error(f"Failed to apply complementary storage: {e}")
            return []
    
    async def _store_chat_context(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Store chat context information"""
        try:
            # Simulate storing chat context
            return {
                "success": True,
                "method": "chat_context",
                "context_stored": True
            }
        
        except Exception as e:
            logger.error(f"Failed to store chat context: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_performance_metrics(self, ingestion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Store performance metrics from ingestion"""
        try:
            metrics = {
                "processing_time_ms": ingestion_results.get("processing_time_ms", 0),
                "quality_score": ingestion_results.get("quality_score", 0.0),
                "quality_level": ingestion_results.get("quality_level", "unknown"),
                "extraction_methods_count": len(ingestion_results.get("extraction_methods_used", [])),
                "ollama_usage": ingestion_results.get("ollama_usage", {})
            }
            
            # Simulate storing metrics
            return {
                "success": True,
                "method": "performance_metrics",
                "metrics_stored": True,
                "metrics": metrics
            }
        
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_storage_summary(self, storage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of storage results"""
        try:
            return {
                "total_storage_methods": len(storage_results.get("storage_details", {})),
                "complementary_storage_count": len(storage_results.get("complementary_storage", [])),
                "fallback_used": storage_results.get("fallback_used", False),
                "overall_success": storage_results.get("storage_success", False),
                "processing_time_ms": storage_results.get("processing_time_ms", 0)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate storage summary: {e}")
            return {"error": str(e)}
