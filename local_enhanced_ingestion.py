"""
Local Enhanced File Ingestion with Ollama Models
Replicates LangExtract capabilities using local Ollama infrastructure
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import time
from datetime import datetime
import aiofiles
import aiohttp
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ExtractionMethod(Enum):
    """Available extraction methods"""
    OLLAMA_STRUCTURED = "ollama_structured"      # Main method using llama3.2:3b
    OLLAMA_EMBEDDINGS = "ollama_embeddings"     # Using mxbai-embed-large
    CURRENT_SYSTEM = "current_system"            # Your existing system
    FALLBACK = "fallback"                        # Basic text processing
    HYBRID = "hybrid"                           # Combine multiple methods

class DocumentType(Enum):
    """Document types for specialized processing"""
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    CONFIG = "config"
    LOG = "log"
    UNKNOWN = "unknown"

class LocalEnhancedIngestionEngine:
    """
    Local enhanced ingestion engine using Ollama models
    Replicates LangExtract capabilities without external APIs
    """
    
    def __init__(self, 
                 ollama_base_url: str = "http://127.0.0.1:11434",
                 current_system: Any = None):
        
        self.ollama_base_url = ollama_base_url
        self.current_system = current_system
        self.session = None
        
        # Ollama models
        self.models = {
            "llm": "llama3.2:3b",
            "embedding": "mxbai-embed-large"
        }
        
        # Quality thresholds
        self.confidence_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "fair": 0.5,
            "poor": 0.3
        }
        
        # Document type detection patterns
        self.file_patterns = {
            DocumentType.CODE: [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".php"],
            DocumentType.MARKDOWN: [".md", ".markdown", ".rst"],
            DocumentType.JSON: [".json", ".jsonl"],
            DocumentType.XML: [".xml", ".html", ".htm", ".svg"],
            DocumentType.CSV: [".csv", ".tsv"],
            DocumentType.CONFIG: [".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"],
            DocumentType.LOG: [".log", ".txt"],
            DocumentType.TEXT: [".txt", ".md", ".rst"]
        }
        
        # Specialized extraction prompts
        self.extraction_prompts = {
            DocumentType.CODE: """Analyze this code file and extract structured information. 
            Return ONLY valid JSON with this exact structure:
            {
                "title": "filename or main class/function name",
                "language": "programming language detected",
                "functions": [
                    {
                        "name": "function name",
                        "params": ["param1", "param2"],
                        "description": "what the function does",
                        "line_start": 10,
                        "line_end": 25
                    }
                ],
                "classes": [
                    {
                        "name": "class name",
                        "methods": ["method1", "method2"],
                        "inheritance": "parent class if any",
                        "line_start": 30,
                        "line_end": 80
                    }
                ],
                "imports": ["import1", "import2"],
                "dependencies": ["dep1", "dep2"],
                "complexity": "low/medium/high",
                "confidence": 0.85
            }
            
            Code content:
            {content}
            """,
            
            DocumentType.MARKDOWN: """Parse this markdown document and extract structured information.
            Return ONLY valid JSON with this exact structure:
            {
                "title": "document title from first heading",
                "headings": [
                    {
                        "level": 1,
                        "text": "heading text",
                        "content": "content under this heading until next heading"
                    }
                ],
                "links": ["url1", "url2"],
                "images": ["image1", "image2"],
                "code_blocks": ["code1", "code2"],
                "tables": [
                    {
                        "headers": ["col1", "col2"],
                        "data": [["row1col1", "row1col2"], ["row2col1", "row2col2"]]
                    }
                ],
                "confidence": 0.9
            }
            
            Markdown content:
            {content}
            """,
            
            DocumentType.JSON: """Analyze this JSON file and extract metadata.
            Return ONLY valid JSON with this exact structure:
            {
                "title": "filename or root key name",
                "type": "object/array/primitive",
                "structure": {
                    "keys": ["key1", "key2"],
                    "nested_levels": 3,
                    "array_lengths": {"key1": 10, "key2": 5}
                },
                "schema": "inferred schema description",
                "size_bytes": 1024,
                "confidence": 0.95
            }
            
            JSON content:
            {content}
            """,
            
            DocumentType.CONFIG: """Parse this configuration file and extract settings.
            Return ONLY valid JSON with this exact structure:
            {
                "title": "config filename or main section",
                "format": "yaml/toml/ini/other",
                "sections": [
                    {
                        "name": "section name",
                        "keys": ["key1", "key2"],
                        "values": {"key1": "value1", "key2": "value2"}
                    }
                ],
                "environment_vars": ["VAR1", "VAR2"],
                "dependencies": ["dep1", "dep2"],
                "confidence": 0.85
            }
            
            Config content:
            {content}
            """,
            
            DocumentType.TEXT: """Analyze this text file and extract information.
            Return ONLY valid JSON with this exact structure:
            {
                "title": "filename or first line if it looks like a title",
                "content_type": "log/documentation/other",
                "sections": [
                    {
                        "heading": "inferred section name or 'Content'",
                        "content": "content text",
                        "line_start": 1,
                        "line_end": 50
                    }
                ],
                "entities": ["entity1", "entity2"],
                "key_value_pairs": {"key1": "value1"},
                "line_count": 100,
                "word_count": 500,
                "confidence": 0.8
            }
            
            Text content:
            {content}
            """
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def ingest_file_enhanced(self, 
                                  file_path: str, 
                                  file_type: str = None,
                                  extraction_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main enhanced ingestion method using local Ollama models
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Auto-detect file type if not provided
        if not file_type:
            file_type = file_path.suffix.lower().lstrip('.')
        
        # Determine document type for specialized processing
        doc_type = self._detect_document_type(file_path, file_type)
        
        logger.info(f"Starting enhanced ingestion for {file_path} (type: {file_type}, doc_type: {doc_type.value})")
        
        # Initialize results container
        results = {
            "file_path": str(file_path),
            "file_type": file_type,
            "document_type": doc_type.value,
            "extraction_methods_used": [],
            "primary_result": None,
            "fallback_result": None,
            "complementary_enhancements": [],
            "quality_score": 0.0,
            "quality_level": "failed",
            "processing_time_ms": 0,
            "errors": [],
            "warnings": [],
            "ollama_usage": {
                "llm_calls": 0,
                "embedding_calls": 0,
                "total_tokens": 0
            }
        }
        
        try:
            # Phase 1: Try Ollama structured extraction (primary method)
            primary_result = await self._extract_with_ollama_structured(
                file_path, doc_type, extraction_options
            )
            
            if primary_result["success"]:
                results["primary_result"] = primary_result
                results["extraction_methods_used"].append("ollama_structured")
                results["ollama_usage"]["llm_calls"] += 1
                logger.info("Primary extraction (Ollama structured) successful")
            else:
                results["errors"].append(f"Ollama structured extraction failed: {primary_result.get('error', 'Unknown error')}")
            
            # Phase 2: Try current system as backup or complementary
            current_result = await self._extract_with_current_system(
                file_path, file_type, extraction_options
            )
            
            if current_result["success"]:
                if not results["primary_result"]:
                    # Use as fallback
                    results["fallback_result"] = current_result
                    results["extraction_methods_used"].append("current_system")
                    logger.info("Using current system as fallback")
                else:
                    # Use as complementary enhancement
                    results["complementary_enhancements"].append(current_result)
                    results["extraction_methods_used"].append("current_system")
                    logger.info("Using current system as complementary enhancement")
            else:
                results["warnings"].append(f"Current system failed: {current_result.get('error', 'Unknown error')}")
            
            # Phase 3: Always try basic fallback for text files
            fallback_result = await self._extract_with_fallback(file_path, file_type)
            
            if fallback_result["success"]:
                if not results["primary_result"] and not results["fallback_result"]:
                    results["fallback_result"] = fallback_result
                    results["extraction_methods_used"].append("fallback")
                    logger.info("Using basic fallback extraction")
            else:
                results["warnings"].append(f"Fallback failed: {fallback_result.get('error', 'Unknown error')}")
            
            # Phase 4: Generate embeddings for content chunks
            if results["primary_result"] or results["fallback_result"]:
                embedding_result = await self._generate_content_embeddings(
                    file_path, results, doc_type
                )
                
                if embedding_result["success"]:
                    results["ollama_usage"]["embedding_calls"] += embedding_result["embedding_calls"]
                    results["ollama_usage"]["total_tokens"] += embedding_result["total_tokens"]
                    results["complementary_enhancements"].append(embedding_result)
                    results["extraction_methods_used"].append("ollama_embeddings")
                    logger.info("Content embeddings generated successfully")
                
                # Combine and enhance results
                enhanced_result = await self._combine_and_enhance_results(results)
                results.update(enhanced_result)
                
                # Calculate quality score
                quality_metrics = self._calculate_quality_score(results)
                results["quality_score"] = quality_metrics["score"]
                results["quality_level"] = quality_metrics["level"]
                
                logger.info(f"Enhanced ingestion completed with quality: {results['quality_level']}")
            else:
                results["errors"].append("All extraction methods failed")
                logger.error("All extraction methods failed")
        
        except Exception as e:
            error_msg = f"Unexpected error during enhanced ingestion: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        finally:
            results["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return results
    
    def _detect_document_type(self, file_path: Path, file_type: str) -> DocumentType:
        """Detect document type based on file extension and content"""
        # Check file extension first
        for doc_type, patterns in self.file_patterns.items():
            if file_type in patterns:
                return doc_type
        
        # Fallback to text for unknown types
        return DocumentType.TEXT
    
    async def _extract_with_ollama_structured(self, 
                                            file_path: Path, 
                                            doc_type: DocumentType,
                                            options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract structured content using Ollama LLM"""
        try:
            logger.info(f"Attempting Ollama structured extraction for {file_path} (type: {doc_type.value})")
            
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            # Get appropriate prompt for document type
            prompt_template = self.extraction_prompts.get(doc_type, self.extraction_prompts[DocumentType.TEXT])
            prompt = prompt_template.format(content=content[:8000])  # Limit content length
            
            # Prepare Ollama request
            payload = {
                "model": self.models["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 2048
                }
            }
            
            # Make request to Ollama
            async with self.session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "response" in data:
                        response_text = data["response"]
                        
                        # Try to extract JSON from response
                        try:
                            # Find JSON content in the response
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            
                            if json_match:
                                json_content = json_match.group(0)
                                extracted_data = json.loads(json_content)
                                
                                # Validate extracted data structure
                                validated_data = self._validate_extracted_data(extracted_data, doc_type)
                                
                                return {
                                    "success": True,
                                    "method": "ollama_structured",
                                    "data": validated_data,
                                    "confidence": validated_data.get("confidence", 0.7),
                                    "raw_response": response_text,
                                    "metadata": {
                                        "model": self.models["llm"],
                                        "prompt_tokens": len(prompt.split()),
                                        "response_tokens": len(response_text.split()),
                                        "total_tokens": data.get("eval_count", 0)
                                    }
                                }
                            else:
                                raise ValueError("No JSON content found in response")
                                
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Failed to parse Ollama JSON response: {e}")
                            # Return structured fallback
                            return {
                                "success": True,
                                "method": "ollama_structured",
                                "data": self._create_fallback_structure(file_path, doc_type, content),
                                "confidence": 0.4,
                                "raw_response": response_text,
                                "warning": f"JSON parsing failed: {e}"
                            }
                    else:
                        return {
                            "success": False,
                            "error": "Invalid response format from Ollama",
                            "response": data
                        }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Ollama API error: {response.status} - {error_text}"
                    }
        
        except Exception as e:
            logger.error(f"Ollama structured extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_extracted_data(self, data: Dict[str, Any], doc_type: DocumentType) -> Dict[str, Any]:
        """Validate and clean extracted data structure"""
        try:
            # Ensure required fields exist
            if "confidence" not in data:
                data["confidence"] = 0.7
            
            if "title" not in data:
                data["title"] = f"Extracted {doc_type.value} document"
            
            # Clean and validate confidence
            try:
                confidence = float(data["confidence"])
                data["confidence"] = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                data["confidence"] = 0.7
            
            # Ensure arrays exist
            for field in ["sections", "functions", "classes", "headings", "entities"]:
                if field not in data or not isinstance(data[field], list):
                    data[field] = []
            
            # Ensure objects exist
            for field in ["key_value_pairs", "metadata", "structure"]:
                if field not in data or not isinstance(data[field], dict):
                    data[field] = {}
            
            return data
        
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                "title": f"Validated {doc_type.value} document",
                "confidence": 0.5,
                "sections": [],
                "entities": [],
                "key_value_pairs": {},
                "metadata": {"validation_error": str(e)}
            }
    
    def _create_fallback_structure(self, file_path: Path, doc_type: DocumentType, content: str) -> Dict[str, Any]:
        """Create fallback structure when JSON parsing fails"""
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        return {
            "title": file_path.name,
            "content_type": doc_type.value,
            "sections": [{
                "heading": "Content",
                "content": content[:1000] + "..." if len(content) > 1000 else content,
                "line_start": 1,
                "line_end": len(lines)
            }],
            "entities": [],
            "key_value_pairs": {},
            "line_count": len(lines),
            "word_count": len(content.split()),
            "confidence": 0.4,
            "fallback": True
        }
    
    async def _extract_with_current_system(self, 
                                          file_path: Path, 
                                          file_type: str,
                                          options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract content using the current system"""
        try:
            logger.info(f"Attempting current system extraction for {file_path}")
            
            if self.current_system:
                # Call your existing ingestion method
                result = await self.current_system.ingest_file(str(file_path), file_type)
                return {
                    "success": True,
                    "method": "current_system",
                    "data": result,
                    "confidence": 0.7,
                    "metadata": {"system": "current_system"}
                }
            else:
                # Simulate current system behavior
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                
                return {
                    "success": True,
                    "method": "current_system",
                    "data": {
                        "title": file_path.name,
                        "content": content,
                        "file_type": file_type,
                        "size_bytes": len(content.encode('utf-8'))
                    },
                    "confidence": 0.6,
                    "metadata": {"system": "current_system_simulated"}
                }
        
        except Exception as e:
            logger.error(f"Current system extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_with_fallback(self, 
                                    file_path: Path, 
                                    file_type: str) -> Dict[str, Any]:
        """Basic fallback extraction for text files"""
        try:
            logger.info(f"Attempting fallback extraction for {file_path}")
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            lines = content.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            return {
                "success": True,
                "method": "fallback",
                "data": {
                    "title": file_path.name,
                    "content": content,
                    "line_count": len(lines),
                    "non_empty_lines": len(non_empty_lines),
                    "file_type": file_type,
                    "size_bytes": len(content.encode('utf-8'))
                },
                "confidence": 0.4,
                "metadata": {"system": "fallback", "basic_processing": True}
            }
        
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_content_embeddings(self, 
                                          file_path: Path, 
                                          results: Dict[str, Any],
                                          doc_type: DocumentType) -> Dict[str, Any]:
        """Generate embeddings for content chunks using Ollama"""
        try:
            logger.info(f"Generating content embeddings for {file_path}")
            
            # Get content to embed
            content_to_embed = ""
            if results["primary_result"]:
                primary_data = results["primary_result"]["data"]
                if "sections" in primary_data:
                    for section in primary_data["sections"][:3]:  # Limit to first 3 sections
                        content_to_embed += section.get("content", "") + " "
                elif "content" in primary_data:
                    content_to_embed = primary_data["content"]
            
            if not content_to_embed and results["fallback_result"]:
                fallback_data = results["fallback_result"]["data"]
                content_to_embed = fallback_data.get("content", "")
            
            if not content_to_embed:
                return {
                    "success": False,
                    "error": "No content available for embedding"
                }
            
            # Prepare embedding request
            payload = {
                "model": self.models["embedding"],
                "prompt": content_to_embed[:4000]  # Limit content length
            }
            
            # Make request to Ollama
            async with self.session.post(
                f"{self.ollama_base_url}/api/embeddings",
                json=payload,
                timeout=30
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "embedding" in data:
                        embedding = data["embedding"]
                        
                        return {
                            "success": True,
                            "method": "ollama_embeddings",
                            "data": {
                                "embedding_dimension": len(embedding),
                                "content_length": len(content_to_embed),
                                "embedding_model": self.models["embedding"]
                            },
                            "embedding": embedding,
                            "embedding_calls": 1,
                            "total_tokens": len(content_to_embed.split()),
                            "metadata": {
                                "model": self.models["embedding"],
                                "embedding_size": len(embedding)
                            }
                        }
                    else:
                        return {
                            "success": False,
                            "error": "No embedding in response",
                            "response": data
                        }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Ollama embedding API error: {response.status} - {error_text}"
                    }
        
        except Exception as e:
            logger.error(f"Content embedding generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _combine_and_enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple extraction methods and enhance them"""
        try:
            combined_data = {
                "title": None,
                "sections": [],
                "entities": [],
                "key_value_pairs": {},
                "metadata": {},
                "confidence": 0.0,
                "extraction_sources": [],
                "embeddings": None
            }
            
            # Combine primary and fallback results
            if results["primary_result"]:
                primary_data = results["primary_result"]["data"]
                combined_data.update(primary_data)
                combined_data["extraction_sources"].append({
                    "method": "ollama_structured",
                    "confidence": results["primary_result"]["confidence"]
                })
            
            if results["fallback_result"]:
                fallback_data = results["fallback_result"]["data"]
                # Merge fallback data intelligently
                if not combined_data["title"] and fallback_data.get("title"):
                    combined_data["title"] = fallback_data["title"]
                
                if not combined_data["sections"] and fallback_data.get("content"):
                    combined_data["sections"].append({
                        "heading": "Content",
                        "content": fallback_data["content"],
                        "line_start": 1,
                        "line_end": fallback_data.get("line_count", 1),
                        "source": "fallback"
                    })
                
                combined_data["extraction_sources"].append({
                    "method": "fallback",
                    "confidence": results["fallback_result"]["confidence"]
                })
            
            # Apply complementary enhancements
            for enhancement in results["complementary_enhancements"]:
                enhancement_data = enhancement["data"]
                
                # Add missing information
                if not combined_data["title"] and enhancement_data.get("title"):
                    combined_data["title"] = enhancement_data["title"]
                
                # Merge sections if they don't conflict
                if enhancement_data.get("sections"):
                    for section in enhancement_data["sections"]:
                        if not any(existing["heading"] == section["heading"] 
                                  for existing in combined_data["sections"]):
                            section["source"] = "complementary"
                            combined_data["sections"].append(section)
                
                # Add embeddings if available
                if enhancement_data.get("embedding"):
                    combined_data["embeddings"] = enhancement_data["embedding"]
                
                combined_data["extraction_sources"].append({
                    "method": enhancement["method"],
                    "confidence": enhancement["confidence"]
                })
            
            # Calculate overall confidence
            if combined_data["extraction_sources"]:
                total_confidence = sum(source["confidence"] for source in combined_data["extraction_sources"])
                combined_data["confidence"] = total_confidence / len(combined_data["extraction_sources"])
            
            # Generate summary
            combined_data["summary"] = {
                "total_sections": len(combined_data["sections"]),
                "total_entities": len(combined_data["entities"]),
                "extraction_methods_count": len(combined_data["extraction_sources"]),
                "combined_confidence": combined_data["confidence"],
                "has_embeddings": combined_data["embeddings"] is not None
            }
            
            return {
                "combined_data": combined_data,
                "enhancement_applied": True
            }
        
        except Exception as e:
            logger.error(f"Failed to combine and enhance results: {e}", exc_info=True)
            return {
                "combined_data": {},
                "enhancement_applied": False,
                "error": str(e)
            }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality score based on multiple factors"""
        try:
            score = 0.0
            factors = []
            
            # Method availability (25% weight)
            method_score = len(results["extraction_methods_used"]) / 4.0  # Max 4 methods
            score += method_score * 0.25
            factors.append(f"method_availability: {method_score:.2f}")
            
            # Primary method success (35% weight)
            if results["primary_result"]:
                primary_confidence = results["primary_result"].get("confidence", 0.0)
                score += primary_confidence * 0.35
                factors.append(f"primary_confidence: {primary_confidence:.2f}")
            
            # Fallback availability (20% weight)
            if results["fallback_result"]:
                fallback_confidence = results["fallback_result"].get("confidence", 0.0)
                score += fallback_confidence * 0.20
                factors.append(f"fallback_confidence: {fallback_confidence:.2f}")
            
            # Complementary enhancements (15% weight)
            if results["complementary_enhancements"]:
                enhancement_score = min(len(results["complementary_enhancements"]) / 3.0, 1.0)
                score += enhancement_score * 0.15
                factors.append(f"enhancements: {enhancement_score:.2f}")
            
            # Embeddings availability (5% weight)
            if any("ollama_embeddings" in enhancement.get("method", "") 
                   for enhancement in results["complementary_enhancements"]):
                score += 0.05
                factors.append("embeddings: 1.00")
            
            # Determine quality level
            if score >= self.confidence_thresholds["excellent"]:
                level = "excellent"
            elif score >= self.confidence_thresholds["good"]:
                level = "good"
            elif score >= self.confidence_thresholds["fair"]:
                level = "fair"
            elif score >= self.confidence_thresholds["poor"]:
                level = "poor"
            else:
                level = "failed"
            
            return {
                "score": round(score, 3),
                "level": level,
                "factors": factors
            }
        
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}", exc_info=True)
            return {
                "score": 0.0,
                "level": "failed",
                "factors": [f"error: {str(e)}"]
            }
