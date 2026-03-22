/* ── GENESIS-WHITEBOARD-VECTOR-SCHEMA — Express Server ───────────── */
/* Vector embedding service for Whiteboard intelligence entries.       */
/* CPU fallback with deterministic pseudo-embeddings.                  */
/* Swaps to NeMo Retriever on GPU for real semantic search.            */

import express from "express";
import { VectorStoreService } from "./services/vector-store.service";

const app = express();
app.use(express.json({ limit: "50mb" }));

const PORT = parseInt(process.env.PORT || "8789", 10);
const store = new VectorStoreService();
const syncIntervalMs = parseInt(process.env.SYNC_INTERVAL_MS || "0", 10);

/* ── Health & State ──────────────────────────────────────────────── */

app.get("/health", (_req, res) => {
  const state = store.getState();
  res.json({
    service: "genesis-whiteboard-vector-schema",
    status: "UP",
    mode: state.mode,
    totalDocuments: state.stats.totalDocuments,
    totalEntries: state.stats.totalEntries,
    dimension: state.stats.dimension,
    model: state.stats.model,
    totalSearches: state.stats.totalSearches,
    uptime: state.uptime,
  });
});

app.get("/state", (_req, res) => {
  res.json(store.getState());
});

app.get("/stats", (_req, res) => {
  res.json(store.getStats());
});

/* ── Configuration ───────────────────────────────────────────────── */

app.get("/config/index", (_req, res) => {
  res.json(store.getIndexConfig());
});

app.get("/config/chunk", (_req, res) => {
  res.json(store.getChunkConfig());
});

app.get("/config/rag", (_req, res) => {
  res.json(store.getRagConfig());
});

/* ── Embedding ───────────────────────────────────────────────────── */

app.post("/embed", async (req, res) => {
  const entry = req.body;
  if (!entry.entryId || !entry.intelligence) {
    return res.status(400).json({ error: "entryId and intelligence required" });
  }
  const result = await store.embed(entry);
  res.status(201).json(result);
});

app.post("/embed/batch", async (req, res) => {
  const entries = req.body.entries || req.body;
  if (!Array.isArray(entries)) return res.status(400).json({ error: "entries[] required" });
  const results = await store.embedBatch(entries);
  res.json({ embedded: results.length, results });
});

app.delete("/embed/:entryId", (req, res) => {
  const removed = store.removeEntry(req.params.entryId);
  res.json({ entryId: req.params.entryId, documentsRemoved: removed });
});

/* ── Search ──────────────────────────────────────────────────────── */

app.post("/search", async (req, res) => {
  const request = req.body;
  if (!request.query && !request.queryEmbedding) {
    return res.status(400).json({ error: "query or queryEmbedding required" });
  }
  const result = await store.search({
    query: request.query || "",
    queryEmbedding: request.queryEmbedding,
    topK: request.topK || 10,
    filters: request.filters,
    minScore: request.minScore,
    rerank: request.rerank,
    includeEmbedding: request.includeEmbedding,
  });
  res.json(result);
});

/* ── RAG ─────────────────────────────────────────────────────────── */

app.post("/rag/context", async (req, res) => {
  const { query, topK, filters } = req.body;
  if (!query) return res.status(400).json({ error: "query required" });
  const context = await store.assembleRagContext(query, topK || 10, filters);
  res.json(context);
});

/* ── Whiteboard Sync ─────────────────────────────────────────────── */

app.post("/sync", async (_req, res) => {
  const result = await store.syncFromWhiteboard();
  res.json(result);
});

/* ── Chunking Preview (test endpoint) ────────────────────────────── */

app.post("/chunk/preview", (req, res) => {
  const entry = req.body;
  if (!entry.intelligence) return res.status(400).json({ error: "intelligence text required" });
  const chunks = (store as any).chunk(entry);
  res.json({ chunks, count: chunks.length });
});

/* ── Start ───────────────────────────────────────────────────────── */

// Periodic Whiteboard sync
if (syncIntervalMs > 0) {
  setInterval(() => store.syncFromWhiteboard(), syncIntervalMs);
  console.log(`[VECTOR] Auto-sync enabled (every ${syncIntervalMs / 1000}s)`);
}

app.listen(PORT, () => {
  const state = store.getState();
  console.log(`[VECTOR-SCHEMA] Listening on port ${PORT}`);
  console.log(`[VECTOR-SCHEMA] Mode: ${state.mode}`);
  console.log(`[VECTOR-SCHEMA] Dimension: ${state.indexConfig.dimension} | Model: ${state.indexConfig.model}`);
  console.log(`[VECTOR-SCHEMA] Distance: ${state.indexConfig.distanceMetric} | Index: ${state.indexConfig.indexType}`);
  console.log(`[VECTOR-SCHEMA] Whiteboard: ${state.whiteboardUrl || "not configured"}`);
});
