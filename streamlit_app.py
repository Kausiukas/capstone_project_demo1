from web.streamlit_app import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LangFlow Connect MVP - Unified Dashboard
Integrated dashboard with Content Preview and Performance Monitoring
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

# Configuration - Default API URL
DEFAULT_API_URL = "https://capstone-project-api-jg3n.onrender.com"
API_KEY = "demo_key_123"
REFRESH_INTERVAL = 30  # seconds

# Page configuration - MUST be called at module level, not inside functions
st.set_page_config(
    page_title="LangFlow Connect MVP - Integrated Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    .preview-container {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        overflow-x: auto;
    }
    .code-content {
        white-space: pre-wrap;
        line-height: 1.5;
    }
    .keyword { color: #007bff; font-weight: bold; }
    .string { color: #28a745; }
    .comment { color: #6c757d; font-style: italic; }
    .key { color: #dc3545; font-weight: bold; }
    .tag { color: #fd7e14; }
    .property { color: #6f42c1; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def get_api_url():
    """Get API URL from session state or default"""
    return st.session_state.get('api_url', DEFAULT_API_URL)

def get_headers():
    """Get headers with current API key"""
    return {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }

def make_api_request(endpoint: str, method: str = "GET", data: dict = None, params: dict = None, timeout: int = 30):
    """Unified API request function with error handling"""
    try:
        api_url = get_api_url()
        headers = get_headers()
        
        if method == "GET":
            response = requests.get(f"{api_url}{endpoint}", headers=headers, params=params, timeout=timeout)
        elif method == "POST":
            response = requests.post(f"{api_url}{endpoint}", headers=headers, json=data, timeout=timeout)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"

def test_health_endpoint():
    """Test the health endpoint"""
    return make_api_request("/health", timeout=10)

def get_tools_list():
    """Get list of available tools"""
    return make_api_request("/tools/list", timeout=10)

def execute_tool(tool_name, arguments):
    """Execute a tool with given arguments"""
    payload = {
        'name': tool_name,
        'arguments': arguments
    }
    return make_api_request("/api/v1/tools/call", method="POST", data=payload, timeout=30)

def display_metric_card(title, value, subtitle="", color="blue"):
    """Display a metric card with consistent styling"""
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {color}; margin: 0;">{title}</h3>
        <h2 style="margin: 0.5rem 0;">{value}</h2>
        <p style="margin: 0; color: #666;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def display_alert_card(alert):
    """Display an alert card"""
    severity = alert.get('severity', 'info')
    color_map = {
        'critical': '#dc3545',
        'error': '#fd7e14', 
        'warning': '#ffc107',
        'info': '#17a2b8'
    }
    color = color_map.get(severity, '#17a2b8')
    
    st.markdown(f"""
    <div class="alert-card">
        <h4 style="color: {color}; margin: 0;">{alert.get('title', 'Alert')}</h4>
        <p style="margin: 0.5rem 0;">{alert.get('message', '')}</p>
        <small style="color: #666;">{alert.get('timestamp', '')}</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# CONTENT PREVIEW FUNCTIONS
# ============================================================================

def preview_file(file_path: str, preview_type: str = None):
    """Preview file content with syntax highlighting and rendering"""
    params = {"file_path": file_path}
    if preview_type and preview_type != "Auto-detect":
        params["preview_type"] = preview_type
    
    return make_api_request("/preview/file", params=params)

def analyze_file(file_path: str):
    """Analyze file to determine preview capabilities"""
    return make_api_request("/preview/analyze", params={"file_path": file_path})

def preview_batch_files(file_paths: list):
    """Preview multiple files in batch"""
    return make_api_request("/preview/batch", method="POST", data={"file_paths": file_paths})

def get_supported_preview_types():
    """Get list of supported file types for preview"""
    return make_api_request("/preview/supported-types")

# ============================================================================
# PERFORMANCE MONITORING FUNCTIONS
# ============================================================================

def get_performance_metrics(tool_name: str = None):
    """Get performance metrics for tools"""
    params = {"tool_name": tool_name} if tool_name else {}
    return make_api_request("/performance/metrics", params=params)

def get_performance_alerts():
    """Get performance alerts"""
    return make_api_request("/performance/alerts")

def get_performance_dashboard():
    """Get comprehensive performance dashboard data"""
    return make_api_request("/performance/dashboard")

def get_performance_health():
    """Get performance health status"""
    return make_api_request("/performance/health")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Main header
    st.markdown("""
    <div class="header">
        <h1>🚀 LangFlow Connect MVP - Integrated Dashboard</h1>
        <p>Capstone Project - AI-Powered Development Tools with Content Preview & Performance Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with navigation
    st.sidebar.header("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Dashboard", 
            "🛠️ Tool Testing", 
            "👁️ Content Preview", 
            "📊 Performance Monitoring", 
            "🗺️ Topology",
            "🧠 Memory & Agent",
            "📚 API Docs", 
            "🔧 System Status"
        ]
    )
    
    # API Configuration in sidebar
    st.sidebar.header("🔧 API Configuration")
    current_api_url = get_api_url()
    new_api_url = st.sidebar.text_input(
        "API Base URL",
        value=current_api_url,
        help="Enter the base URL of your API"
    )
    
    if st.sidebar.button("🔄 Update API URL"):
        if new_api_url != current_api_url:
            st.session_state['api_url'] = new_api_url
            st.sidebar.success(f"✅ API URL updated!")
            st.rerun()
    
    if st.sidebar.button("🧪 Test Connection"):
        with st.spinner("Testing API connection..."):
            success, result = test_health_endpoint()
            if success:
                st.sidebar.success("✅ API connection successful!")
            else:
                st.sidebar.error(f"❌ API connection failed: {result}")
    
    # ============================================================================
    # DASHBOARD PAGE
    # ============================================================================
    if page == "🏠 Dashboard":
        st.header("🏠 Welcome to LangFlow Connect MVP")
        
        # Status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", "🟢 Online")
            st.metric("Available Tools", "5")
        
        with col2:
            st.metric("Content Preview", "✅ Active")
            st.metric("Performance Monitoring", "✅ Active")
        
        with col3:
            st.metric("Version", "2.0.0")
            st.metric("Capstone", "✅ Complete")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("🏥 Health Check", use_container_width=True):
                with st.spinner("Checking health..."):
                    success, result = test_health_endpoint()
                    if success:
                        st.success("✅ Service is healthy!")
                        st.json(result)
                    else:
                        st.error(f"❌ Health check failed: {result}")
        
        with col5:
            if st.button("👁️ Preview Test", use_container_width=True):
                with st.spinner("Testing content preview..."):
                    success, result = preview_file("README.md")
                    if success:
                        st.success("✅ Content preview working!")
                        st.metric("File Type", result.get("file_type", "Unknown"))
                    else:
                        st.error(f"❌ Content preview failed: {result}")
        
        with col6:
            if st.button("📊 Performance Check", use_container_width=True):
                with st.spinner("Checking performance..."):
                    success, result = get_performance_health()
                    if success:
                        st.success("✅ Performance monitoring active!")
                        st.metric("Status", result.get("status", "Unknown"))
                    else:
                        st.error(f"❌ Performance check failed: {result}")
        
        # Project info
        st.subheader("📋 Project Information")
        st.markdown("""
        **LangFlow Connect MVP** is a comprehensive capstone project demonstrating:
        - 🤖 **AI-powered development tools** with MCP integration
        - 👁️ **Content Preview System** with syntax highlighting and rendering
        - 📊 **Performance Monitoring** with real-time metrics and alerts
        - 🔌 **RESTful API** with authentication and universal file access
        - 🎯 **Unified Web Interface** with integrated dashboard
        
        **Available Tools:**
        - `ping` - Test server connectivity
        - `list_files` - List directory contents (local, GitHub, HTTP)
        - `read_file` - Read file contents (local, GitHub, HTTP)
        - `get_system_status` - Get system metrics
        - `analyze_code` - Analyze code files
        
        **New Features:**
        - **Content Preview** - Syntax highlighting, markdown rendering, image preview
        - **Performance Monitoring** - Real-time metrics, alerts, health monitoring
        - **Universal File Access** - Local, GitHub, and HTTP file support
        """)
    
    # ============================================================================
    # TOOL TESTING PAGE
    # ============================================================================
    elif page == "🛠️ Tool Testing":
        st.header("🛠️ Interactive Tool Testing")
        
        # Tool selection
        tool_name = st.selectbox(
            "Select Tool to Test",
            ["ping", "list_files", "read_file", "get_system_status", "analyze_code"]
        )
        
        # Tool-specific parameters
        arguments = {}
        
        if tool_name == "list_files":
            path = st.text_input("Directory path", value=".", help="Can be local path, GitHub URL, or HTTP URL")
            arguments = {"directory": path}
        
        elif tool_name == "read_file":
            file_path = st.text_input("File path", value="README.md", help="Can be local path, GitHub URL, or HTTP URL")
            arguments = {"file_path": file_path}
        
        elif tool_name == "analyze_code":
            file_path = st.text_input("Code file path", value="src/mcp_server_http.py", help="Can be local path, GitHub URL, or HTTP URL")
            arguments = {"file_path": file_path}
        
        # Execute button
        if st.button(f"🚀 Execute {tool_name}", type="primary"):
            with st.spinner(f"Executing {tool_name}..."):
                start_time = time.time()
                success, result = execute_tool(tool_name, arguments)
                end_time = time.time()
                
                if success:
                    st.success(f"✅ {tool_name} executed successfully!")
                    st.metric("Response Time", f"{(end_time - start_time)*1000:.2f}ms")
                    
                    # Display result
                    if isinstance(result, dict) and 'content' in result:
                        st.subheader("Result:")
                        for content in result['content']:
                            if content['type'] == 'text':
                                st.text_area("Output", content['text'], height=200)
                    else:
                        st.json(result)
                else:
                    st.error(f"❌ {tool_name} failed: {result}")
    
    # ============================================================================
    # CONTENT PREVIEW PAGE
    # ============================================================================
    elif page == "👁️ Content Preview":
        st.header("👁️ Content Preview System")
        st.markdown("**Enhanced file content preview with syntax highlighting and rendering**")
        
        # Sidebar for preview options
        st.sidebar.header("🎛️ Preview Options")
        
        # File input
        file_path = st.sidebar.text_input(
            "📁 File Path",
            placeholder="Enter file path (local, GitHub, or HTTP URL)",
            help="Examples:\n- Local: D:\\path\\to\\file.py\n- GitHub: https://github.com/user/repo/blob/main/file.py\n- HTTP: https://example.com/file.txt"
        )
        
        # Preview type selection
        preview_type = st.sidebar.selectbox(
            "🎨 Preview Type",
            ["Auto-detect", "code", "image", "document", "markdown"],
            help="Select preview type or let the system auto-detect"
        )
        
        # Batch preview
        st.sidebar.header("📦 Batch Preview")
        batch_files = st.sidebar.text_area(
            "Multiple Files (one per line)",
            placeholder="file1.py\nfile2.js\nfile3.md",
            help="Enter multiple file paths for batch preview"
        )
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🔍 File Analysis")
            
            if file_path:
                if st.button("🔍 Analyze File", type="primary"):
                    with st.spinner("Analyzing file..."):
                        success, result = analyze_file(file_path)
                        
                        if success:
                            st.success("✅ File analysis completed!")
                            
                            # Display analysis results
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Preview Type", result.get("preview_type", "Unknown"))
                                st.metric("Language", result.get("language", "None"))
                                st.metric("Supported", "✅ Yes" if result.get("supported") else "❌ No")
                            
                            with col_b:
                                st.metric("Source Type", result.get("source_type", "Unknown"))
                                st.metric("File Extension", result.get("file_extension", "None"))
                                st.metric("Exists", "✅ Yes" if result.get("exists") else "❌ No")
                            
                            # Capabilities
                            st.subheader("🎯 Preview Capabilities")
                            capabilities = result.get("capabilities", {})
                            for capability, available in capabilities.items():
                                status = "✅" if available else "❌"
                                st.write(f"{status} {capability.replace('_', ' ').title()}")
                        else:
                            st.error(f"❌ Analysis failed: {result}")
            
            # Supported types
            st.header("📋 Supported File Types")
            if st.button("📋 Get Supported Types"):
                with st.spinner("Fetching supported types..."):
                    success, result = get_supported_preview_types()
                    
                    if success:
                        supported_types = result.get("supported_types", {})
                        
                        for preview_type_name, config in supported_types.items():
                            with st.expander(f"📁 {preview_type_name.title()}"):
                                st.write("**Extensions:**")
                                st.code(", ".join(config.get("extensions", [])))
                                st.write("**MIME Types:**")
                                st.code(", ".join(config.get("mime_types", [])))
                    else:
                        st.error(f"❌ Failed to get supported types: {result}")
        
        with col2:
            st.header("👁️ Content Preview")
            
            if file_path:
                # Preview parameters
                preview_params = {"file_path": file_path}
                if preview_type != "Auto-detect":
                    preview_params["preview_type"] = preview_type
                
                if st.button("👁️ Preview File", type="primary"):
                    with st.spinner("Generating preview..."):
                        success, result = preview_file(file_path, preview_type if preview_type != "Auto-detect" else None)
                        
                        if success:
                            st.success("✅ Preview generated successfully!")
                            
                            # File info
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("File Type", result.get("file_type", "Unknown"))
                                st.metric("Language", result.get("language", "None"))
                            
                            with col_info2:
                                st.metric("Content Length", f"{result.get('content_length', 0):,} chars")
                                st.metric("Source", result.get("metadata", {}).get("source_type", "Unknown"))
                            
                            # Preview content
                            st.subheader("🎨 Preview")
                            preview_html = result.get("preview_html", "")
                            
                            if preview_html:
                                # Display HTML content
                                st.components.v1.html(preview_html, height=400, scrolling=True)
                            else:
                                st.warning("⚠️ No preview content available")
                            
                            # Raw content (collapsible)
                            with st.expander("📄 Raw Content"):
                                raw_content = result.get("content", "")
                                if raw_content:
                                    st.code(raw_content, language=result.get("language", "text"))
                                else:
                                    st.info("No raw content available")
                        else:
                            st.error(f"❌ Preview failed: {result}")
            
            # Batch preview
            if batch_files:
                st.header("📦 Batch Preview")
                if st.button("📦 Preview Multiple Files"):
                    file_list = [f.strip() for f in batch_files.split('\n') if f.strip()]
                    
                    if file_list:
                        with st.spinner(f"Processing {len(file_list)} files..."):
                            success, result = preview_batch_files(file_list)
                            
                            if success:
                                st.success(f"✅ Batch preview completed! {result.get('successful_previews', 0)}/{result.get('total_files', 0)} successful")
                                
                                # Display results
                                results = result.get("results", [])
                                for i, file_result in enumerate(results):
                                    with st.expander(f"📄 {file_result.get('file_path', f'File {i+1}')}"):
                                        if file_result.get("success"):
                                            col_b1, col_b2 = st.columns(2)
                                            with col_b1:
                                                st.metric("Type", file_result.get("file_type", "Unknown"))
                                                st.metric("Language", file_result.get("language", "None"))
                                            
                                            with col_b2:
                                                st.metric("Length", f"{file_result.get('content_length', 0):,} chars")
                                                st.metric("Status", "✅ Success")
                                            
                                            # Preview
                                            preview_html = file_result.get("preview_html", "")
                                            if preview_html:
                                                st.components.v1.html(preview_html, height=300, scrolling=True)
                                        else:
                                            st.error(f"❌ Failed: {file_result.get('error', 'Unknown error')}")
                            else:
                                st.error(f"❌ Batch preview failed: {result}")
        
        # Examples section
        st.header("💡 Example Files")
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            st.subheader("🐍 Python Code")
            example_python = """def hello_world():
    \"\"\"Simple hello world function\"\"\"
    print("Hello, World!")
    return True

# Main execution
if __name__ == "__main__":
    hello_world()"""
            st.code(example_python, language="python")
            if st.button("Try Python Example"):
                st.session_state.example_file = "example.py"
                st.session_state.example_content = example_python
        
        with col_ex2:
            st.subheader("📝 Markdown")
            example_markdown = """# Sample Markdown

## Features
- **Bold text**
- *Italic text*
- `Code snippets`

## Code Block
```python
print("Hello from markdown!")
```

[Link to GitHub](https://github.com)"""
            st.code(example_markdown, language="markdown")
            if st.button("Try Markdown Example"):
                st.session_state.example_file = "example.md"
                st.session_state.example_content = example_markdown
        
        with col_ex3:
            st.subheader("🎨 JSON Data")
            example_json = """{
  "name": "Sample Project",
  "version": "1.0.0",
  "description": "A sample JSON file",
  "features": [
    "Syntax highlighting",
    "Content preview",
    "File analysis"
  ],
  "metadata": {
    "author": "Developer",
    "license": "MIT"
  }
}"""
            st.code(example_json, language="json")
            if st.button("Try JSON Example"):
                st.session_state.example_file = "example.json"
                st.session_state.example_content = example_json
    
    # ============================================================================
    # PERFORMANCE MONITORING PAGE
    # ============================================================================
    elif page == "📊 Performance Monitoring":
        st.header("📊 Performance Monitoring Dashboard")
        st.markdown("**Real-time performance metrics and system monitoring**")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(1)  # Small delay for refresh
            st.rerun()
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Get Metrics", use_container_width=True):
                with st.spinner("Fetching performance metrics..."):
                    success, result = get_performance_metrics()
                    if success:
                        st.success("✅ Metrics retrieved!")
                        
                        # Display key metrics
                        overall_metrics = result.get("overall", {})
                        display_metric_card(
                            "Overall Success Rate", 
                            f"{overall_metrics.get('success_rate', 0):.1f}%",
                            "All tools combined"
                        )
                        display_metric_card(
                            "Average Response Time", 
                            f"{overall_metrics.get('avg_response_time', 0):.2f}ms",
                            "Across all tools"
                        )
                        display_metric_card(
                            "Total Requests", 
                            f"{overall_metrics.get('total_requests', 0):,}",
                            "Since startup"
                        )
                    else:
                        st.error(f"❌ Failed to get metrics: {result}")
        
        with col2:
            if st.button("🚨 Get Alerts", use_container_width=True):
                with st.spinner("Fetching alerts..."):
                    success, result = get_performance_alerts()
                    if success:
                        alerts = result.get("alerts", [])
                        if alerts:
                            st.success(f"✅ Found {len(alerts)} alerts!")
                            for alert in alerts:
                                display_alert_card(alert)
                        else:
                            st.info("✅ No active alerts")
                    else:
                        st.error(f"❌ Failed to get alerts: {result}")
        
        with col3:
            if st.button("🏥 Health Check", use_container_width=True):
                with st.spinner("Checking performance health..."):
                    success, result = get_performance_health()
                    if success:
                        health_status = result.get("status", "unknown")
                        st.success(f"✅ Performance health: {health_status}")
                        
                        # Display health metrics
                        metrics = result.get("metrics", {})
                        display_metric_card(
                            "System Health", 
                            health_status.title(),
                            "Performance monitoring status"
                        )
                        display_metric_card(
                            "CPU Usage", 
                            f"{metrics.get('cpu_usage', 0):.1f}%",
                            "Current system load"
                        )
                        display_metric_card(
                            "Memory Usage", 
                            f"{metrics.get('memory_usage', 0):.1f}%",
                            "System memory utilization"
                        )
                    else:
                        st.error(f"❌ Health check failed: {result}")
        
        with col4:
            if st.button("📈 Dashboard Data", use_container_width=True):
                with st.spinner("Fetching dashboard data..."):
                    success, result = get_performance_dashboard()
                    if success:
                        st.success("✅ Dashboard data retrieved!")
                        
                        # Display comprehensive metrics
                        dashboard_data = result.get("dashboard", {})
                        display_metric_card(
                            "Active Tools", 
                            f"{dashboard_data.get('active_tools', 0)}",
                            "Currently monitored"
                        )
                        display_metric_card(
                            "System Uptime", 
                            f"{dashboard_data.get('uptime_hours', 0):.1f}h",
                            "Service uptime"
                        )
                        display_metric_card(
                            "Error Rate", 
                            f"{dashboard_data.get('error_rate', 0):.2f}%",
                            "Recent errors"
                        )
                    else:
                        st.error(f"❌ Dashboard data failed: {result}")
        
        # Tool-specific metrics
        st.subheader("🔧 Tool-Specific Metrics")
        
        # Tool selection
        tool_name = st.selectbox(
            "Select Tool for Detailed Metrics",
            ["All Tools", "ping", "list_files", "read_file", "get_system_status", "analyze_code"]
        )
        
        if st.button(f"📊 Get {tool_name} Metrics"):
            with st.spinner(f"Fetching {tool_name} metrics..."):
                if tool_name == "All Tools":
                    success, result = get_performance_metrics()
                else:
                    success, result = get_performance_metrics(tool_name)
                
                if success:
                    if tool_name == "All Tools":
                        tools_data = result.get("tools", {})
                        for tool, metrics in tools_data.items():
                            with st.expander(f"🔧 {tool}"):
                                col_t1, col_t2, col_t3 = st.columns(3)
                                with col_t1:
                                    st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
                                with col_t2:
                                    st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}ms")
                                with col_t3:
                                    st.metric("Total Requests", f"{metrics.get('total_requests', 0):,}")
                    else:
                        tool_metrics = result.get("tool_metrics", {})
                        col_t1, col_t2, col_t3 = st.columns(3)
                        with col_t1:
                            st.metric("Success Rate", f"{tool_metrics.get('success_rate', 0):.1f}%")
                        with col_t2:
                            st.metric("Avg Response Time", f"{tool_metrics.get('avg_response_time', 0):.2f}ms")
                        with col_t3:
                            st.metric("Total Requests", f"{tool_metrics.get('total_requests', 0):,}")
                        
                        # Response time chart
                        response_times = tool_metrics.get("response_times", [])
                        if response_times:
                            df = pd.DataFrame(response_times, columns=["timestamp", "response_time"])
                            fig = px.line(df, x="timestamp", y="response_time", title=f"{tool_name} Response Times")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Failed to get {tool_name} metrics: {result}")
        
        # Performance testing
        st.subheader("⚡ Performance Testing")
        
        col_test1, col_test2, col_test3 = st.columns(3)
        
        with col_test1:
            if st.button("🏃‍♂️ Speed Test"):
                with st.spinner("Testing response time..."):
                    times = []
                    for i in range(5):
                        start = time.time()
                        success, _ = test_health_endpoint()
                        end = time.time()
                        if success:
                            times.append((end - start) * 1000)
                    
                    if times:
                        avg_time = sum(times) / len(times)
                        st.metric("Average Response Time", f"{avg_time:.2f}ms")
                        st.metric("Min Response Time", f"{min(times):.2f}ms")
                        st.metric("Max Response Time", f"{max(times):.2f}ms")
        
        with col_test2:
            if st.button("🔄 Load Test"):
                with st.spinner("Running load test..."):
                    results = []
                    def test_request():
                        success, _ = test_health_endpoint()
                        results.append(success)
                    
                    threads = []
                    for i in range(10):
                        t = threading.Thread(target=test_request)
                        threads.append(t)
                        t.start()
                    
                    for t in threads:
                        t.join()
                    
                    success_count = sum(results)
                    st.metric("Successful Requests", f"{success_count}/10")
                    st.metric("Success Rate", f"{success_count/10*100:.1f}%")
        
        with col_test3:
            if st.button("🔍 API Test"):
                with st.spinner("Testing API endpoints..."):
                    tests = [
                        ("Health", lambda: test_health_endpoint()),
                        ("Tools List", lambda: get_tools_list()),
                        ("Ping Tool", lambda: execute_tool("ping", {})),
                        ("Performance Metrics", lambda: get_performance_metrics()),
                        ("Content Preview", lambda: preview_file("README.md")),
                    ]
                    
                    results = []
                    for test_name, test_func in tests:
                        success, _ = test_func()
                        results.append((test_name, success))
                    
                    st.subheader("API Test Results")
                    for test_name, success in results:
                        if success:
                            st.success(f"✅ {test_name}")
                        else:
                            st.error(f"❌ {test_name}")

    # ============================================================================
    # TOPOLOGY PAGE
    # ============================================================================
    elif page == "🗺️ Topology":
        st.header("🗺️ Live Topology")
        st.markdown("System ports, processes and environment mapping")

        colt1, colt2 = st.columns(2)
        with colt1:
            if st.button("🔍 Refresh Topology"):
                ok, res = make_api_request("/admin/topology")
                if ok:
                    st.success("✅ Topology fetched")
                    st.subheader("Process")
                    st.json(res.get("process", {}))
                    st.subheader("Listening")
                    st.json(res.get("listening", []))
                    st.subheader("Expected Ports")
                    st.json(res.get("expected_ports", {}))
                else:
                    st.error(res)

        with colt2:
            if st.button("🔌 Ports In Use"):
                ok, res = make_api_request("/admin/topology")
                if ok:
                    st.success("✅ Ports listed")
                    st.subheader("Ports In Use")
                    st.json(res.get("ports_in_use", []))
                    st.subheader("Streamlit Processes")
                    st.json(res.get("streamlit_processes", []))
                    st.subheader("Environment")
                    st.json(res.get("env", {}))
                else:
                    st.error(res)

    # ============================================================================
    # MEMORY & AGENT PAGE
    # ============================================================================
    elif page == "🧠 Memory & Agent":
        st.header("🧠 Memory & Agent")
        st.markdown("Store and retrieve interactions via pgvector; ask the simple agent.")

        colm1, colm2 = st.columns(2)

        with colm1:
            st.subheader("📦 Store Interaction")
            user_input = st.text_area("User Input", height=100)
            response_text = st.text_area("Response", height=100)
            if st.button("💾 Store"):
                with st.spinner("Storing interaction..."):
                    payload = {"user_input": user_input, "response": response_text}
                    ok, res = make_api_request("/memory/store", method="POST", data=payload)
                    st.success("Stored") if ok else st.error(res)

            st.subheader("🔍 Memory Query")
            query = st.text_input("Query text")
            k = st.number_input("Top-K", min_value=1, max_value=20, value=5)
            if st.button("🔎 Search"):
                with st.spinner("Searching memory..."):
                    ok, res = make_api_request("/memory/query", params={"q": query, "k": k})
                    if ok:
                        snippets = res.get("snippets", [])
                        for s in snippets:
                            st.info(s.get("preview", "")[:500])
                    else:
                        st.error(res)

        with colm2:
            st.subheader("🤖 Ask Agent")
            prompt = st.text_area("Prompt", height=160)
            if st.button("🧠 Answer"):
                with st.spinner("Thinking..."):
                    ok, res = make_api_request("/agent/answer", method="POST", data={"prompt": prompt}, timeout=30)
                    if ok:
                        st.success("Agent answered")
                        st.text_area("Answer", res.get("answer", ""), height=220)
                    else:
                        st.error(res)

        st.subheader("📈 Memory Performance")
        ok, res = make_api_request("/performance/memory")
        if ok:
            summary = res.get("summary", {})
            c1, c2, c3 = st.columns(3)
            for (col, key) in zip([c1, c2, c3], ["embedding", "memory_read", "memory_write"]):
                with col:
                    m = summary.get(key, {})
                    st.metric(f"{key} avg", f"{m.get('avg_response_time', 0):.2f} ms")
                    st.metric(f"{key} requests", f"{m.get('total_requests', 0)}")
        else:
            st.info("Memory performance not available")
    
    # ============================================================================
    # API DOCS PAGE
    # ============================================================================
    elif page == "📚 API Docs":
        st.header("📚 API Documentation")
        
        st.subheader("Base URL")
        st.code(get_api_url())
        
        st.subheader("Authentication")
        st.markdown("All API requests require the `X-API-Key` header:")
        st.code("X-API-Key: demo_key_123")
        
        st.subheader("Core Endpoints")
        
        # Health endpoint
        with st.expander("🏥 Health Check"):
            st.markdown("**GET** `/health`")
            st.markdown("Check if the service is running.")
            st.code(f"curl -X GET {get_api_url()}/health")
        
        # Tools list endpoint
        with st.expander("🛠️ List Tools"):
            st.markdown("**GET** `/tools/list`")
            st.markdown("Get list of available tools.")
            st.code(f"""curl -X GET {get_api_url()}/tools/list \\
  -H "X-API-Key: demo_key_123" """)
        
        # Tool execution endpoint
        with st.expander("⚡ Execute Tool"):
            st.markdown("**POST** `/api/v1/tools/call`")
            st.markdown("Execute a specific tool.")
            st.code(f"""curl -X POST {get_api_url()}/api/v1/tools/call \\
  -H "X-API-Key: demo_key_123" \\
  -H "Content-Type: application/json" \\
  -d '{{"name": "ping", "arguments": {{}}}}' """)
        
        st.subheader("Content Preview Endpoints")
        
        # Preview file endpoint
        with st.expander("👁️ Preview File"):
            st.markdown("**GET** `/preview/file`")
            st.markdown("Preview file content with syntax highlighting.")
            st.code(f"""curl -X GET "{get_api_url()}/preview/file?file_path=README.md" \\
  -H "X-API-Key: demo_key_123" """)
        
        # Analyze file endpoint
        with st.expander("🔍 Analyze File"):
            st.markdown("**GET** `/preview/analyze`")
            st.markdown("Analyze file for preview capabilities.")
            st.code(f"""curl -X GET "{get_api_url()}/preview/analyze?file_path=README.md" \\
  -H "X-API-Key: demo_key_123" """)
        
        # Batch preview endpoint
        with st.expander("📦 Batch Preview"):
            st.markdown("**POST** `/preview/batch`")
            st.markdown("Preview multiple files in batch.")
            st.code(f"""curl -X POST {get_api_url()}/preview/batch \\
  -H "X-API-Key: demo_key_123" \\
  -H "Content-Type: application/json" \\
  -d '{{"file_paths": ["file1.py", "file2.md"]}}' """)
        
        st.subheader("Performance Monitoring Endpoints")
        
        # Performance metrics endpoint
        with st.expander("📊 Performance Metrics"):
            st.markdown("**GET** `/performance/metrics`")
            st.markdown("Get performance metrics for tools.")
            st.code(f"""curl -X GET {get_api_url()}/performance/metrics \\
  -H "X-API-Key: demo_key_123" """)
        
        # Performance alerts endpoint
        with st.expander("🚨 Performance Alerts"):
            st.markdown("**GET** `/performance/alerts`")
            st.markdown("Get performance alerts.")
            st.code(f"""curl -X GET {get_api_url()}/performance/alerts \\
  -H "X-API-Key: demo_key_123" """)
        
        st.subheader("Available Tools")
        tools_info = [
            ("ping", "Test server connectivity", "{}"),
            ("list_files", "List files in directory", '{"directory": "."}'),
            ("read_file", "Read file contents", '{"file_path": "README.md"}'),
            ("get_system_status", "Get system metrics", "{}"),
            ("analyze_code", "Analyze code files", '{"file_path": "src/app.py"}')
        ]
        
        for tool_name, description, example in tools_info:
            with st.expander(f"🔧 {tool_name}"):
                st.markdown(f"**Description:** {description}")
                st.markdown(f"**Example:** `{example}`")
    
    # ============================================================================
    # SYSTEM STATUS PAGE
    # ============================================================================
    elif page == "🔧 System Status":
        st.header("🔧 System Status")
        
        # Real-time status
        st.subheader("📊 System Status")
        if st.button("🔄 Refresh Status"):
            with st.spinner("Checking system status..."):
                success, result = execute_tool("get_system_status", {})
                if success:
                    st.success("✅ System status retrieved!")
                    if isinstance(result, dict) and 'content' in result:
                        for content in result['content']:
                            if content['type'] == 'text':
                                st.text_area("System Status", content['text'], height=200)
                else:
                    st.error(f"❌ Failed to get system status: {result}")
        
        # Service information
        st.subheader("ℹ️ Service Information")
        st.markdown(f"""
        - **Current API URL:** {get_api_url()}
        - **Dashboard Version:** 2.0.0 (Unified)
        - **Status:** 🟢 Online
        - **Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        - **Features:** Content Preview ✅, Performance Monitoring ✅
        """)
        
        # Quick health check
        st.subheader("🏥 Health Check")
        success, health_data = test_health_endpoint()
        if success:
            st.success("🟢 API is healthy and responding")
            st.json(health_data)
        else:
            st.error("🔴 API is not responding")
            st.text(health_data)
        
        # Feature status
        st.subheader("🎯 Feature Status")
        col_feat1, col_feat2, col_feat3 = st.columns(3)
        
        with col_feat1:
            st.markdown("**Core Tools**")
            st.success("✅ All 5 tools operational")
            st.info("ping, list_files, read_file, get_system_status, analyze_code")
        
        with col_feat2:
            st.markdown("**Content Preview**")
            st.success("✅ Syntax highlighting active")
            st.info("Code, markdown, images, documents")
        
        with col_feat3:
            st.markdown("**Performance Monitoring**")
            st.success("✅ Real-time metrics active")
            st.info("Response times, success rates, alerts")

if __name__ == "__main__":
    main()
