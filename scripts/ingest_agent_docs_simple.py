#!/usr/bin/env python3
"""
Simple Agent Documentation Ingestion Script

This script ingests all the essential documentation files directly via the API
to avoid import conflicts and module dependencies.
"""

import os
import requests
from pathlib import Path
from datetime import datetime

# Configuration
API_BASE = "http://127.0.0.1:8000"
API_KEY = "demo_key_123"

# Documentation files to ingest (in order of importance)
DOCS_TO_INGEST = [
    # Core Architecture
    {
        "path": "docs/Identifyable_build.md",
        "title": "Identifiable API Builds - Docker Identity, PGVector, Ollama",
        "category": "architecture",
        "priority": 1
    },
    {
        "path": "docs/identificator.md", 
        "title": "API Instance Identification System",
        "category": "architecture",
        "priority": 1
    },
    {
        "path": "docs/ping_trace.md",
        "title": "Smart Ping and Tracing System",
        "category": "architecture", 
        "priority": 1
    },
    
    # Agent Layers Structure
    {
        "path": "src/layers/goals.py",
        "title": "Agent Goals and Objectives System",
        "category": "layers",
        "priority": 2
    },
    {
        "path": "src/layers/memory_system.py",
        "title": "PostgreSQL Vector Memory System",
        "category": "layers",
        "priority": 2
    },
    {
        "path": "src/layers/llm_adapter.py",
        "title": "LLM Adapter Layer",
        "category": "layers",
        "priority": 2
    },
    {
        "path": "src/layers/tool_orchestrator.py",
        "title": "Tool Orchestration Layer",
        "category": "layers",
        "priority": 2
    },
    
    # Implementation Details
    {
        "path": "src/modules/module_2_support/postgresql_vector_agent.py",
        "title": "PostgreSQL Vector Agent Implementation",
        "category": "implementation",
        "priority": 3
    },
    
    # Configuration and Deployment
    {
        "path": "deployment/docker/docker-compose.yml",
        "title": "Docker Compose Configuration",
        "category": "deployment",
        "priority": 3
    },
    
    # Goals Configuration
    {
        "path": "repo_capstone_project_demo1/7_agent_layers/LVL_3/goals.yaml",
        "title": "Agent Goals Configuration YAML",
        "category": "goals",
        "priority": 2
    }
]

def ingest_documentation():
    """Ingest all documentation into the PostgreSQL vector database via API"""
    
    print("🚀 Starting Agent Documentation Ingestion...")
    print(f"📊 Target: {len(DOCS_TO_INGEST)} documentation files")
    print(f"🌐 API Base: {API_BASE}")
    print()
    
    # Track ingestion results
    successful = 0
    failed = 0
    skipped = 0
    
    for doc_info in DOCS_TO_INGEST:
        file_path = Path(doc_info["path"])
        
        print(f"📖 Processing: {doc_info['title']}")
        print(f"   📁 Path: {file_path}")
        print(f"   🏷️  Category: {doc_info['category']}")
        print(f"   ⭐ Priority: {doc_info['priority']}")
        
        # Check if file exists
        if not file_path.exists():
            print(f"   ⚠️  File not found, skipping...")
            skipped += 1
            print()
            continue
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            if not content.strip():
                print(f"   ⚠️  File is empty, skipping...")
                skipped += 1
                print()
                continue
            
            # Store via API
            try:
                response = requests.post(
                    f"{API_BASE}/memory/store",
                    headers={"X-API-Key": API_KEY},
                    json={
                        "user_input": f"documentation: {doc_info['title']}",
                        "response": content
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        print(f"   ✅ Successfully ingested (ID: {result.get('record_id', 'N/A')})")
                        print(f"   🗄️  Type: {result.get('type', 'N/A')}")
                        successful += 1
                    else:
                        print(f"   ❌ Storage failed: {result.get('error')}")
                        failed += 1
                else:
                    print(f"   ❌ API request failed: {response.status_code}")
                    print(f"   📝 Response: {response.text[:200]}...")
                    failed += 1
                    
            except Exception as e:
                print(f"   ❌ Error during API call: {e}")
                failed += 1
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            failed += 1
        
        print()
    
    # Print summary
    print("📊 Ingestion Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Skipped: {skipped}")
    print(f"   📁 Total Processed: {successful + failed + skipped}")
    
    if successful > 0:
        print(f"\n🎉 Successfully ingested {successful} documentation files!")
        print("🤖 The agent now has comprehensive knowledge of its own structure.")
        
        # Test retrieval
        print("\n🧪 Testing retrieval capabilities...")
        try:
            test_query = "What is the agent's memory system architecture?"
            test_response = requests.get(
                f"{API_BASE}/memory/query",
                headers={"X-API-Key": API_KEY},
                params={"query": test_query, "k": 3}
            )
            
            if test_response.status_code == 200:
                result = test_response.json()
                if result.get("success"):
                    print(f"   ✅ Retrieval test successful: {result.get('total_found')} results found")
                    print(f"   🗄️  Database type: {result.get('type')}")
                    
                    # Show sample results
                    results = result.get("results", [])
                    if results:
                        print(f"   📋 Sample results:")
                        for i, res in enumerate(results[:2]):
                            print(f"      {i+1}. {res.get('preview', '')[:100]}...")
                else:
                    print(f"   ❌ Retrieval test failed: {result.get('error')}")
            else:
                print(f"   ❌ Retrieval test failed: {test_response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Retrieval test error: {e}")
    
    return successful > 0

def main():
    """Main entry point"""
    try:
        success = ingest_documentation()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Ingestion interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
