# Advanced Multi-Stage RAG Platform with Dynamic Knowledge Base Ingestion

An enterprise-ready, context-aware Retrieval-Augmented Generation (RAG) assistant built utilizing **Streamlit**, **LangChain**, and **FAISS**, powered by the production-ready **Google Gemini API** (`gemini-1.5-flash`). 

This platform features an advanced asynchronous incremental data indexing pipeline designed to expand the system's operational knowledge base dynamically via multi-format FAQ data tables (.csv) without requiring server restarts or complete index recalculations.

---

## 🛠️ System Architecture & Data Flow

```text
    [Uploaded Bulk CSV Data] ──> [Pandas Ingestion Stream (Latin1 Fallback)]
                                                 │
                                                 ▼
    [FAISS Local Vector Store]  <── [Atomic Index Merge] <── [Document Object Generation]
                │
                ▼
    [Semantic Query Match] ──> [Context Extraction Filter] ──> [Gemini Inference Engine]