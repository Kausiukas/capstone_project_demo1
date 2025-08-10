# 🧭 Layer 2: Information Gathering & Context

### **Function & Purpose**
Collect, normalize, and synthesize information from files, systems, HTTP, and external sources to supply high‑quality context to reasoning and tools.

### **Core Components**
- **Collectors**: Filesystem, HTTP, system metrics, logs
- **Normalizers**: Type detection, parsing, preview extraction
- **Synthesizer**: Merge multi‑source snippets into briefs
- **Validators**: Quality, freshness, provenance checks

### **Current Implementation Status** ✅ **80% Complete**

#### **✅ What's Implemented**
- File preview and analysis
- System status and performance metrics
- Topology endpoint and diagnostics scripts
- Content type detection and safe rendering

#### **❌ What's Missing**
- Web/data feed collectors and schedulers
- Context synthesis (multi‑source briefs)
- Confidence/freshness scoring and provenance
- External API collectors with backoff/limits

### **Implementation Plan**

#### **Phase 1: Enhanced Collection (Week 1-2)**
- Add HTTP fetch collector with content sniffing
- Add simple schedule loop for periodic scraping
- Store parsed snippets to Memory (pgvector) with tags

#### **Phase 2: Synthesis (Week 3-4)**
- Merge source snippets into concise briefs (<1 KB)
- Attach provenance and freshness scores; persist

#### **Phase 3: Validation (Week 5-6)**
- Deduplicate and stale‑marking; TTL policies
- Integrity checks and per‑source health

---

## 🧪 Operational Notes
- Respect robots.txt and rate limits; implement exponential backoff
- Tag all context with `source_type`, `path/url`, and `collected_at`

## 🎯 Next 48h Priorities
- Ship HTTP collector + schedule loop
- Add synthesis brief writer and tests
- Index newly collected snippets via `PostgreSQLVectorAgent`
