# 🧠 Layer 6: Memory & Feedback

### **Function & Purpose**
Store episodic knowledge and feedback; retrieve relevant context; learn from outcomes.

### **Core Components**
- **Short‑term Memory**: Conversation window + autosummary
- **Long‑term Memory**: pgvector store (embeddings, metadata)
- **Feedback Loop**: Up/down signals; ranking adjustments
- **Consolidation**: Cluster→summarize→TTL/decay

### **Current Implementation Status** ⚠️ **30% Complete**

#### **✅ What's Implemented**
- `MemorySystem` with store/query and feedback endpoints
- pgvector agent with parameterized inserts/searches and dim auto‑table
- Basic tests for store/search and feedback ranking

#### **❌ What's Missing**
- Short‑term window + autosummary compressor
- Consolidation jobs and TTL/decay
- HNSW/metric toggles and performance tuning

### **Implementation Plan**

#### **Phase 1: Short‑term + Ranking (Week 1-2)**
- Rolling window; autosummary; feedback‑weighted ranking

#### **Phase 2: Consolidation (Week 3-4)**
- Nightly cluster→summarize; archive originals; TTL policies

#### **Phase 3: Performance (Week 5-6)**
- HNSW index; metric toggle; retrieval optimizer

---

## 🧪 Operational Notes
- Redact PII in stored content; support purge by tag/user

## 🎯 Next 48h Priorities
- Implement short‑term window + autosummary
- Add consolidation job skeleton and tests
