#!/usr/bin/env python3
"""
Test script to check database connection and schema creation
"""

import asyncio
import asyncpg
import os
from typing import Dict, Any

async def test_database_connection():
    """Test database connection and schema creation"""
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL", "postgresql://langflow_user:langflow_password@localhost:5432/langflow_connect")
    
    print(f"Testing database connection with: {database_url}")
    
    try:
        # Test basic connection
        print("1. Testing basic connection...")
        conn = await asyncpg.connect(database_url)
        print("✅ Basic connection successful")
        
        # Test pgvector extension
        print("2. Testing pgvector extension...")
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"❌ Failed to enable pgvector extension: {e}")
            return False
        
        # Test vector_store table creation
        print("3. Testing vector_store table creation...")
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_store (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source VARCHAR(500) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1024),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            print("✅ vector_store table created successfully")
        except Exception as e:
            print(f"❌ Failed to create vector_store table: {e}")
            return False
        
        # Test chat_history table creation
        print("4. Testing chat_history table creation...")
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    context_docs JSONB DEFAULT '[]',
                    used_docs INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    response_time_ms INTEGER,
                    model_used VARCHAR(100),
                    streamed BOOLEAN DEFAULT FALSE
                )
            """)
            print("✅ chat_history table created successfully")
        except Exception as e:
            print(f"❌ Failed to create chat_history table: {e}")
            return False
        
        # Test table existence
        print("5. Testing table existence...")
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('vector_store', 'chat_history')
        """)
        
        existing_tables = [row['table_name'] for row in tables]
        print(f"✅ Existing tables: {existing_tables}")
        
        # Test inserting a test document
        print("6. Testing document insertion...")
        try:
            test_id = await conn.fetchval("""
                INSERT INTO vector_store (source, content) 
                VALUES ($1, $2) 
                RETURNING id
            """, "test_file.txt", "This is a test document for testing the database schema.")
            print(f"✅ Test document inserted with ID: {test_id}")
            
            # Clean up test document
            await conn.execute("DELETE FROM vector_store WHERE id = $1", test_id)
            print("✅ Test document cleaned up")
            
        except Exception as e:
            print(f"❌ Failed to insert test document: {e}")
            return False
        
        await conn.close()
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def main():
    """Main function"""
    print("🔍 Database Schema Test")
    print("=" * 50)
    
    success = await test_database_connection()
    
    if success:
        print("\n✅ Database is ready for enhanced ingestion!")
    else:
        print("\n❌ Database setup failed. Please check the errors above.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
