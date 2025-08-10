# LangFlow Connect MVP - Integrated Dashboard

## 🎯 Project Overview

This is a **Capstone Project** demonstration of the LangFlow Connect MCP (Model Context Protocol) server with integrated Content Preview and Performance Monitoring systems.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run MCP server
python src/mcp_server_enhanced_tools.py

# Run integrated dashboard
streamlit run streamlit_app_integrated.py
```
## PostgreSQL + pgvector (Docker)

Prereqs:
- Docker Desktop on Windows with WSL2 backend enabled

Start/stop:
- Start DB: `docker compose up -d`
- Stop DB: `docker compose down`
- Logs: `docker compose logs -f db`
- Verify: `python scripts/verify_pgvector.py`

Automation:

```bash
# Ensure DB is up, wait for readiness, ensure pgvector, and print DATABASE_URL
python scripts/ensure_pgvector.py --timeout 120

# Optionally set it in the current process and see the value
python scripts/ensure_pgvector.py --set-env

# Then run tests that need the DB
pytest -q tests/phase1/test_pgvector_basic.py
```

Details:
- Data persists in the `pgdata` Docker volume
- Config via `.env` (defaults provided in docs); connection string `DATABASE_URL`
- Default port mapping `5432:5432`. If port is in use, change left side to `5433:5432` and set `DATABASE_URL=postgresql://app:app@localhost:5433/appdb`

Troubleshooting:
- Port 5432 in use → change mapping and `DATABASE_URL`
- WSL high memory → `wsl --shutdown` in PowerShell to reclaim RAM
- Resource limits → Docker Desktop → Settings → Resources


## 🛠 Features

### Core Tools
- **5 Core Tools**: ping, read_file, list_files, get_system_status, analyze_code
- **Universal File Access**: Local, GitHub, and HTTP file support
- **Web Interface**: Streamlit-based unified dashboard
- **API Access**: RESTful API for programmatic access

### Content Preview System
- **Syntax Highlighting**: Support for 20+ programming languages
- **Markdown Rendering**: Full markdown to HTML conversion
- **Image Preview**: Base64 encoding for inline display
- **Batch Processing**: Preview multiple files simultaneously
- **File Analysis**: Automatic type detection and capabilities

### Performance Monitoring
- **Real-time Metrics**: Response times, success rates, error counts
- **System Monitoring**: CPU, memory, disk usage tracking
- **Performance Alerts**: Automated alerting for issues
- **Health Monitoring**: Comprehensive system health checks
- **Tool-specific Metrics**: Individual tool performance tracking

## 📊 Dashboard Sections

1. **🏠 Dashboard** - Overview and quick actions
2. **🛠️ Tool Testing** - Interactive tool execution
3. **👁️ Content Preview** - File preview and analysis
4. **📊 Performance Monitoring** - Real-time metrics and alerts
5. **🗺️ Topology** - Live ports/processes/env and DB/pgvector status
5. **📚 API Docs** - Complete API documentation
6. **🔧 System Status** - System health and configuration

## 🔧 Configuration

The dashboard automatically connects to the deployed API at:
`https://capstone-project-api-jg3n.onrender.com`

You can change the API URL in the sidebar configuration.

## 🧰 Utilities

- `scripts/ports_status.py`: list listening ports and owning processes; helpful for resolving conflicts (8000/8501/5433)
- `scripts/find_free_port.py`: find an available local TCP port in a range
- `scripts/topology_probe.py`: query the API `/admin/topology` endpoint and pretty-print results
- `scripts/warmup_api.py`: warm up key endpoints (`/health`, `/performance/*`, `/db/*`) to prime caches/DB

Examples:

```bash
python scripts/ports_status.py --common
python scripts/find_free_port.py --start 8000 --end 8100
python scripts/topology_probe.py --api http://127.0.0.1:8000 --key demo_key_123
python scripts/warmup_api.py --api http://127.0.0.1:8000 --key demo_key_123
```

## 📄 License

MIT License

## 🎯 Capstone Project Status

✅ **Complete** - All systems integrated and functional
- Core MCP tools operational
- Content Preview System active
- Performance Monitoring active
- Unified dashboard deployed
- Universal file access working
- Real-time metrics collection
- Comprehensive error handling
