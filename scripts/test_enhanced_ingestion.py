#!/usr/bin/env python3
"""
Test Script for Enhanced Ingestion System
Tests the local Ollama-based enhanced ingestion capabilities
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedIngestionTester:
    """Test the enhanced ingestion system"""
    
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000", api_key: str = "demo_key_123"):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_system_status(self) -> Dict[str, Any]:
        """Test the enhanced ingestion system status"""
        try:
            logger.info("Testing enhanced ingestion system status...")
            
            async with self.session.get(
                f"{self.api_base_url}/ingest/enhanced/status",
                headers=self.headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Enhanced ingestion status retrieved successfully")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to get status: HTTP {response.status} - {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"❌ Status test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_enhanced_ingestion(self, file_path: str) -> Dict[str, Any]:
        """Test enhanced ingestion with a file"""
        try:
            logger.info(f"Testing enhanced ingestion with file: {file_path}")
            
            # Check if file exists
            if not Path(file_path).exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            # Prepare file for upload
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Create multipart form data
            data = aiohttp.FormData()
            data.add_field('file', file_data, filename=Path(file_path).name)
            
            # Make request
            async with self.session.post(
                f"{self.api_base_url}/ingest/enhanced",
                data=data,
                headers=self.headers
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ Enhanced ingestion successful")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Enhanced ingestion failed: HTTP {response.status} - {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"❌ Enhanced ingestion test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_batch_ingestion(self, file_paths: List[str]) -> Dict[str, Any]:
        """Test batch enhanced ingestion"""
        try:
            logger.info(f"Testing batch enhanced ingestion with {len(file_paths)} files...")
            
            # Prepare files for upload
            data = aiohttp.FormData()
            for file_path in file_paths:
                if Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    data.add_field('files', file_data, filename=Path(file_path).name)
                else:
                    logger.warning(f"File not found: {file_path}")
            
            # Make request
            async with self.session.post(
                f"{self.api_base_url}/ingest/enhanced/batch",
                data=data,
                headers=self.headers
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ Batch enhanced ingestion successful")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Batch enhanced ingestion failed: HTTP {response.status} - {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"❌ Batch enhanced ingestion test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_current_system_compatibility(self) -> Dict[str, Any]:
        """Test compatibility with current system"""
        try:
            logger.info("Testing current system compatibility...")
            
            # Test existing endpoints
            endpoints_to_test = [
                "/health",
                "/memory/documents",
                "/memory/query"
            ]
            
            results = {}
            for endpoint in endpoints_to_test:
                try:
                    async with self.session.get(
                        f"{self.api_base_url}{endpoint}",
                        headers=self.headers
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            results[endpoint] = {"success": True, "data": data}
                        else:
                            results[endpoint] = {"success": False, "status": response.status}
                
                except Exception as e:
                    results[endpoint] = {"success": False, "error": str(e)}
            
            logger.info("✅ Current system compatibility test completed")
            return {"success": True, "endpoint_tests": results}
        
        except Exception as e:
            logger.error(f"❌ Current system compatibility test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        logger.info("🚀 Starting Enhanced Ingestion Comprehensive Test Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        test_results = {}
        
        try:
            # Test 1: System Status
            logger.info("\n🔍 Test 1: Enhanced Ingestion System Status")
            status_result = await self.test_system_status()
            test_results["system_status"] = status_result
            
            if not status_result.get("success"):
                logger.error("❌ System status test failed, stopping tests")
                return test_results
            
            # Test 2: Current System Compatibility
            logger.info("\n🔍 Test 2: Current System Compatibility")
            compatibility_result = await self.test_current_system_compatibility()
            test_results["current_system_compatibility"] = compatibility_result
            
            # Test 3: Enhanced Ingestion (if we have test files)
            test_files = self._find_test_files()
            if test_files:
                logger.info(f"\n🔍 Test 3: Enhanced Ingestion with {len(test_files)} test files")
                
                for test_file in test_files[:3]:  # Test first 3 files
                    logger.info(f"  Testing file: {test_file}")
                    ingestion_result = await self.test_enhanced_ingestion(test_file)
                    test_results[f"ingestion_{Path(test_file).name}"] = ingestion_result
                    
                    # Wait a bit between files
                    await asyncio.sleep(1)
                
                # Test batch ingestion if we have multiple files
                if len(test_files) > 1:
                    logger.info(f"\n🔍 Test 4: Batch Enhanced Ingestion")
                    batch_result = await self.test_batch_ingestion(test_files[:3])
                    test_results["batch_ingestion"] = batch_result
            else:
                logger.warning("⚠️  No test files found, skipping ingestion tests")
                test_results["ingestion_tests"] = {"skipped": "No test files available"}
            
            # Generate summary
            total_tests = len(test_results)
            successful_tests = sum(1 for result in test_results.values() 
                                 if isinstance(result, dict) and result.get("success"))
            
            test_results["summary"] = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 COMPREHENSIVE TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Successful: {successful_tests}")
            logger.info(f"Failed: {total_tests - successful_tests}")
            logger.info(f"Execution Time: {test_results['summary']['execution_time_ms']}ms")
            
            if successful_tests == total_tests:
                logger.info("🎉 All tests passed! Enhanced ingestion system is working correctly.")
            else:
                logger.warning("⚠️  Some tests failed. Check the results for details.")
            
            return test_results
        
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            test_results["comprehensive_test"] = {"success": False, "error": str(e)}
            return test_results
    
    def _find_test_files(self) -> List[str]:
        """Find test files in the current directory"""
        test_files = []
        current_dir = Path(".")
        
        # Look for common file types
        file_patterns = [
            "*.txt", "*.md", "*.py", "*.js", "*.json", "*.xml", "*.csv",
            "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg", "*.conf"
        ]
        
        for pattern in file_patterns:
            test_files.extend([str(f) for f in current_dir.glob(pattern)])
        
        # Filter out very large files and common system files
        filtered_files = []
        for file_path in test_files:
            try:
                file_size = Path(file_path).stat().st_size
                if file_size < 1024 * 1024:  # Less than 1MB
                    filtered_files.append(file_path)
            except:
                continue
        
        return filtered_files[:10]  # Limit to 10 files

async def main():
    """Main test function"""
    print("🔧 Enhanced Ingestion System Tester")
    print("=" * 50)
    
    # Configuration
    api_url = "http://127.0.0.1:8000"
    api_key = "demo_key_123"
    
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key}")
    print()
    
    # Run tests
    async with EnhancedIngestionTester(api_url, api_key) as tester:
        results = await tester.run_comprehensive_test()
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_ingestion_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Test results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to save results: {e}")
        
        return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        exit_code = 0 if results.get("summary", {}).get("failed_tests", 1) == 0 else 1
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        exit(1)
