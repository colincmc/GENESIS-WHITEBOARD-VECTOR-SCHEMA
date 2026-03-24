# GENESIS-WHITEBOARD-VECTOR-SCHEMA

**RAG-ready vector embeddings — semantic search over Whiteboard institutional memory**

**Port:** `8789`

> **NVIDIA Phase 2A** — This service is a GPU-readiness pipe. It defines schemas, formats, and CPU-side logic that will activate when NVIDIA RAPIDS / NeMo / Warp hardware arrives. Phase 0 runs pure TypeScript on CPU.

---

## What It Does

1. Embeds Whiteboard intelligence documents into vector space using deterministic hash-based pseudo-embeddings (CPU fallback for GPU embedding models)
2. Provides semantic search via brute-force cosine similarity with metadata filtering (category, source, confidence, rail, tags)
3. Assembles RAG (Retrieval-Augmented Generation) context from search results with token counting, DOCTRINE score boosting, and configurable system prompts
4. Supports 4 chunking strategies: SINGLE (whole document), SENTENCE (period-split), SLIDING_WINDOW (configurable overlap), SEMANTIC (paragraph-based)
5. Syncs with Whiteboard via HTTP pull — fetches and embeds intelligence entries on demand or on a timer
6. Deduplicates entries by source document ID to prevent index bloat
7. Boosts DOCTRINE-tagged documents in search results (1.5x score multiplier) per Genesis intelligence hierarchy

---

## Architecture

| File | Purpose | Lines |
|------|---------|-------|
| `src/index.ts` | Express server — 13 endpoints for embed, search, RAG context, Whiteboard sync, config | 138 |
| `src/types.ts` | VectorIndexConfig (HNSW/IVF_PQ/FLAT), ChunkConfig (4 strategies), VectorDocument, VectorSearchRequest, RagContext, RagConfig | 222 |
| `src/services/vector-store.service.ts` | Core engine — chunking, CPU hash embeddings (LCG + L2 norm), cosine similarity search, metadata filtering, DOCTRINE boosting, RAG assembly, Whiteboard sync | 539 |
| `package.json` | Express dependency | 18 |
| `Dockerfile` | node:20.20.0-slim, EXPOSE 8789 | 10 |

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service health + index stats |
| GET | `/state` | Full store state (document count, index config, embedding model) |
| POST | `/embed` | Embed a single document (text + metadata) |
| POST | `/embed/batch` | Embed an array of documents |
| DELETE | `/embed/:id` | Remove a document from the index |
| POST | `/search` | Semantic vector search with metadata filters |
| POST | `/rag/context` | Assemble RAG context from search results (with token limit) |
| POST | `/sync/whiteboard` | Pull and embed entries from Whiteboard |
| GET | `/chunks/preview` | Preview chunking output for a given text |
| GET | `/config/index` | Vector index configuration (dimension, model, index type) |
| GET | `/config/chunk` | Chunking configuration (strategy, window size, overlap) |
| GET | `/config/rag` | RAG configuration (max tokens, doctrine boost, system prompt) |
| GET | `/stats` | Index statistics (document count, embedding count, search count) |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8789` | HTTP listen port |
| `EMBEDDING_DIMENSION` | `768` | Vector embedding dimension |
| `EMBEDDING_MODEL` | `ALL_MINILM` | Embedding model name (CPU fallback uses hash embeddings) |
| `GPU_EMBED_URL` | `null` | GPU embedding service URL (replaces CPU hash embeddings) |
| `WHITEBOARD_URL` | `null` | Whiteboard URL for sync operations |
| `SYNC_INTERVAL_MS` | `0` | Auto-sync interval (0 = disabled, manual sync only) |
| `MAX_DOCUMENTS` | `50000` | Maximum documents in the vector index |

---

## Integration

- **Reads from:** Whiteboard (/entries endpoint — intelligence documents), any service posting to /embed
- **Writes to:** Internal vector index (in-memory), console (sync/search results)
- **Consumed by:** CIA (RAG context for strategic analysis), NeMo Guardrails (grounded inference), Brighton Protocol (pattern similarity search)
- **GPU future:** GPU embedding models (sentence-transformers) replace CPU hash embeddings, HNSW/IVF_PQ GPU-accelerated index (Phase 2+)

---

## Current State

- **Phase 0 BUILT** — CPU-side TypeScript, fully operational
- Deterministic hash-based pseudo-embeddings (LCG with L2 normalisation) as GPU model placeholder
- Brute-force cosine similarity search with metadata filtering
- 4 chunking strategies (SINGLE, SENTENCE, SLIDING_WINDOW, SEMANTIC)
- DOCTRINE score boosting (1.5x multiplier)
- RAG context assembly with token counting
- Whiteboard sync via HTTP pull
- Entry deduplication by source document ID

---

## Future Editions

1. **GPU embedding models** — replace hash embeddings with sentence-transformers (all-MiniLM-L6-v2) on GPU via TensorRT
2. **HNSW GPU index** — swap brute-force search for GPU-accelerated HNSW approximate nearest neighbour (FAISS/RAFT)
3. **IVF_PQ quantisation** — product quantisation for million-document scale with minimal memory
4. **NeMo Retriever integration** — plug into NeMo Retriever pipeline for enterprise RAG with guardrails
5. **Cross-rail vector search** — federated search across Rail 1 (crypto), Rail 3 (forex), Rail 5 (gold) Whiteboards

---

## Rail Deployment

| Rail | Status | Notes |
|------|--------|-------|
| Rail 1 | BUILT | CPU hash embeddings, cosine similarity, DOCTRINE boosting, RAG assembly |
| Rail 3 | GPU activation | TensorRT embedding models, FAISS/RAFT GPU index |
| Rail 5+ | Full NVIDIA stack | NeMo Retriever, IVF_PQ quantisation, cross-rail federated search |
