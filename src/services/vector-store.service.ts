/* ── GENESIS-WHITEBOARD-VECTOR-SCHEMA — Vector Store Service ─────── */
/* In-memory vector store with cosine similarity search.              */
/* CPU fallback using brute-force. Swaps to FAISS/Milvus on GPU.     */
/* Handles chunking, embedding dispatch, search, RAG assembly.        */

import { randomUUID } from "crypto";
import {
  VectorDocument,
  VectorMetadata,
  VectorSearchRequest,
  VectorSearchResult,
  VectorMatch,
  RagContext,
  EmbedRequest,
  EmbedResponse,
  VectorIndexConfig,
  ChunkConfig,
  RagConfig,
  VectorIndexStats,
  VectorServiceState,
  ConfidenceLevel,
  IntelCategory,
  IntelSource,
  EmbeddingModel,
} from "../types";

/* ── Default Configurations ──────────────────────────────────────── */

const DEFAULT_INDEX_CONFIG: VectorIndexConfig = {
  dimension: parseInt(process.env.EMBEDDING_DIMENSION || "768", 10),
  distanceMetric: "COSINE",
  indexType: "HNSW",
  model: (process.env.EMBEDDING_MODEL as EmbeddingModel) || "ALL_MINILM",
  hnswM: 16,
  hnswEfConstruction: 200,
  hnswEfSearch: 100,
  ivfNlist: 100,
  ivfNprobe: 10,
  pqM: 8,
  pqBits: 8,
};

const DEFAULT_CHUNK_CONFIG: ChunkConfig = {
  strategy: "SINGLE",
  maxTokens: 256,
  overlapTokens: 32,
  minTokens: 16,
  includeMetadata: true,
  metadataTemplate: "[{category}] [{source}] [{confidence}]: ",
};

const DEFAULT_RAG_CONFIG: RagConfig = {
  maxContextTokens: 2048,
  contextTemplate: "--- Intel #{rank} (confidence: {confidence}, category: {category}) ---\n{text}\n",
  systemPrompt: "You are a Genesis intelligence analyst. Answer queries using ONLY the retrieved Whiteboard intelligence below. Cite confidence levels. Flag any DOCTRINE-level findings prominently.",
  includeMetadata: true,
  deduplicateEntries: true,
  boostDoctrine: true,
  doctrineBoostFactor: 1.5,
};

export class VectorStoreService {
  private documents: Map<string, VectorDocument> = new Map();
  private indexConfig: VectorIndexConfig;
  private chunkConfig: ChunkConfig;
  private ragConfig: RagConfig;
  private totalSearches = 0;
  private searchTimes: number[] = [];
  private lastEmbeddedAt: string | null = null;
  private lastSearchAt: string | null = null;
  private gpuEmbedUrl: string | null;
  private whiteboardUrl: string | null;
  private startedAt = Date.now();
  private maxDocuments = parseInt(process.env.MAX_DOCUMENTS || "50000", 10);

  constructor() {
    this.indexConfig = { ...DEFAULT_INDEX_CONFIG };
    this.chunkConfig = { ...DEFAULT_CHUNK_CONFIG };
    this.ragConfig = { ...DEFAULT_RAG_CONFIG };
    this.gpuEmbedUrl = process.env.GPU_EMBED_URL || null;
    this.whiteboardUrl = process.env.WHITEBOARD_URL || null;

    // Adjust dimension based on model
    if (this.indexConfig.model === "ALL_MINILM") this.indexConfig.dimension = 384;
    else if (this.indexConfig.model === "E5_LARGE" || this.indexConfig.model === "BGE_LARGE") this.indexConfig.dimension = 1024;
    else if (this.indexConfig.model === "NEMOTRON_MINI") this.indexConfig.dimension = 768;
  }

  /* ── Chunking ──────────────────────────────────────────────────── */

  chunk(entry: EmbedRequest): string[] {
    const chunks: string[] = [];
    const metaPrefix = this.chunkConfig.includeMetadata
      ? this.chunkConfig.metadataTemplate
          .replace("{category}", entry.category)
          .replace("{source}", entry.source)
          .replace("{confidence}", entry.confidenceLevel)
      : "";

    // Primary intelligence text
    const primaryText = metaPrefix + entry.intelligence;

    if (this.chunkConfig.strategy === "SINGLE") {
      chunks.push(primaryText);
    } else if (this.chunkConfig.strategy === "SENTENCE") {
      const sentences = entry.intelligence.split(/[.!?]+/).filter(s => s.trim().length > 0);
      let current = metaPrefix;
      for (const sentence of sentences) {
        const trimmed = sentence.trim();
        if (this.estimateTokens(current + trimmed) > this.chunkConfig.maxTokens && current.length > metaPrefix.length) {
          chunks.push(current.trim());
          current = metaPrefix;
        }
        current += trimmed + ". ";
      }
      if (current.length > metaPrefix.length) chunks.push(current.trim());
    } else if (this.chunkConfig.strategy === "SLIDING_WINDOW") {
      const words = entry.intelligence.split(/\s+/);
      const windowSize = Math.floor(this.chunkConfig.maxTokens * 0.75);
      const step = windowSize - this.chunkConfig.overlapTokens;
      for (let i = 0; i < words.length; i += Math.max(1, step)) {
        const windowWords = words.slice(i, i + windowSize);
        if (windowWords.length >= this.chunkConfig.minTokens) {
          chunks.push(metaPrefix + windowWords.join(" "));
        }
      }
    } else {
      // SEMANTIC — fall back to single for CPU mode
      chunks.push(primaryText);
    }

    // Evidence as additional chunks (if present and long)
    if (entry.evidence && entry.evidence.length > 0) {
      const evidenceText = metaPrefix + "Evidence: " + entry.evidence.join("; ");
      if (this.estimateTokens(evidenceText) <= this.chunkConfig.maxTokens) {
        chunks.push(evidenceText);
      }
    }

    return chunks.length > 0 ? chunks : [primaryText];
  }

  /* ── Embedding ─────────────────────────────────────────────────── */

  async embed(entry: EmbedRequest): Promise<EmbedResponse> {
    const start = Date.now();
    const chunks = this.chunk(entry);
    const documents: VectorDocument[] = [];

    const metadata: VectorMetadata = {
      category: entry.category,
      source: entry.source,
      confidenceLevel: entry.confidenceLevel,
      confidence: entry.confidence,
      affectedRails: entry.affectedRails,
      affectedClasses: entry.affectedClasses,
      tags: entry.tags,
      firstSeen: entry.firstSeen,
      lastSeen: entry.lastSeen,
      observations: entry.observations,
      active: true,
    };

    for (let i = 0; i < chunks.length; i++) {
      const embedding = await this.getEmbedding(chunks[i]);

      const doc: VectorDocument = {
        documentId: randomUUID(),
        entryId: entry.entryId,
        chunkIndex: i,
        totalChunks: chunks.length,
        text: chunks[i],
        embedding,
        metadata,
        embeddedAt: new Date().toISOString(),
        model: this.indexConfig.model,
      };

      this.documents.set(doc.documentId, doc);
      documents.push(doc);
    }

    this.lastEmbeddedAt = new Date().toISOString();
    this.enforceLimit();

    return {
      entryId: entry.entryId,
      documents,
      chunksCreated: chunks.length,
      embeddingTimeMs: Date.now() - start,
    };
  }

  async embedBatch(entries: EmbedRequest[]): Promise<EmbedResponse[]> {
    const results: EmbedResponse[] = [];
    for (const entry of entries) {
      results.push(await this.embed(entry));
    }
    return results;
  }

  /* ── Vector Similarity Search ──────────────────────────────────── */

  async search(request: VectorSearchRequest): Promise<VectorSearchResult> {
    const start = Date.now();
    this.totalSearches++;

    // Get query embedding
    const queryEmbedding = request.queryEmbedding || await this.getEmbedding(request.query);
    const topK = request.topK || 10;
    const minScore = request.minScore ?? 0.5;

    // Filter candidates by metadata
    let candidates = Array.from(this.documents.values());
    if (request.filters) {
      candidates = this.applyFilters(candidates, request.filters);
    }

    // Compute similarity scores
    let scored: Array<{ doc: VectorDocument; score: number }> = [];
    for (const doc of candidates) {
      const score = this.cosineSimilarity(queryEmbedding, doc.embedding);
      if (score >= minScore) {
        scored.push({ doc, score });
      }
    }

    // Boost DOCTRINE entries
    if (this.ragConfig.boostDoctrine) {
      scored = scored.map(s => ({
        ...s,
        score: s.doc.metadata.confidenceLevel === "DOCTRINE"
          ? Math.min(1, s.score * this.ragConfig.doctrineBoostFactor)
          : s.score,
      }));
    }

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    // Deduplicate by entryId (keep highest scoring chunk)
    if (this.ragConfig.deduplicateEntries) {
      const seen = new Set<string>();
      scored = scored.filter(s => {
        if (seen.has(s.doc.entryId)) return false;
        seen.add(s.doc.entryId);
        return true;
      });
    }

    // Take top K
    const topResults = scored.slice(0, topK);

    const results: VectorMatch[] = topResults.map(s => ({
      documentId: s.doc.documentId,
      entryId: s.doc.entryId,
      chunkIndex: s.doc.chunkIndex,
      text: s.doc.text,
      score: parseFloat(s.score.toFixed(4)),
      metadata: s.doc.metadata,
      embedding: request.includeEmbedding ? s.doc.embedding : undefined,
    }));

    const elapsed = Date.now() - start;
    this.searchTimes.push(elapsed);
    if (this.searchTimes.length > 100) this.searchTimes.shift();
    this.lastSearchAt = new Date().toISOString();

    return {
      searchId: randomUUID(),
      query: request.query,
      results,
      totalFound: results.length,
      searchTimeMs: elapsed,
      model: this.indexConfig.model,
      reranked: false,
    };
  }

  /* ── RAG Context Assembly ──────────────────────────────────────── */

  async assembleRagContext(query: string, topK = 10, filters?: VectorSearchRequest["filters"]): Promise<RagContext> {
    const searchResult = await this.search({
      query,
      topK,
      filters,
      minScore: 0.3,
    });

    let formattedContext = "";
    let tokenCount = 0;
    let truncated = false;

    for (let i = 0; i < searchResult.results.length; i++) {
      const match = searchResult.results[i];
      const formatted = this.ragConfig.contextTemplate
        .replace("{rank}", String(i + 1))
        .replace("{confidence}", match.metadata.confidenceLevel)
        .replace("{category}", match.metadata.category)
        .replace("{text}", match.text);

      const chunkTokens = this.estimateTokens(formatted);
      if (tokenCount + chunkTokens > this.ragConfig.maxContextTokens) {
        truncated = true;
        break;
      }

      formattedContext += formatted + "\n";
      tokenCount += chunkTokens;
    }

    return {
      contextId: randomUUID(),
      query,
      retrievedDocuments: searchResult.results,
      formattedContext,
      tokenCount,
      maxTokens: this.ragConfig.maxContextTokens,
      truncated,
      assembledAt: new Date().toISOString(),
    };
  }

  /* ── Whiteboard Sync ───────────────────────────────────────────── */

  async syncFromWhiteboard(): Promise<{ embedded: number; errors: number }> {
    if (!this.whiteboardUrl) return { embedded: 0, errors: 0 };

    let embedded = 0;
    let errors = 0;

    try {
      // Fetch all intel from Whiteboard
      const res = await fetch(`${this.whiteboardUrl}/intel/search?limit=5000`);
      if (!res.ok) return { embedded: 0, errors: 1 };
      const data = (await res.json()) as any;
      const entries: any[] = Array.isArray(data) ? data : (data.entries || data.results || []);

      for (const entry of entries) {
        try {
          await this.embed({
            entryId: entry.entryId,
            intelligence: entry.intelligence,
            evidence: entry.evidence,
            category: entry.category,
            source: entry.source,
            confidenceLevel: entry.confidenceLevel,
            confidence: entry.confidence,
            affectedRails: entry.affectedRails || [],
            affectedClasses: entry.affectedClasses || [],
            tags: entry.tags || [],
            firstSeen: entry.firstSeen,
            lastSeen: entry.lastSeen,
            observations: entry.observations,
          });
          embedded++;
        } catch {
          errors++;
        }
      }

      console.log(`[VECTOR] Synced ${embedded} entries from Whiteboard (${errors} errors)`);
    } catch {
      console.warn("[VECTOR] Failed to sync from Whiteboard");
      errors++;
    }

    return { embedded, errors };
  }

  /* ── CPU Embedding (Deterministic Hash-Based Fallback) ─────────── */
  /* Real embeddings come from NeMo Retriever on GPU.                 */
  /* CPU mode uses deterministic pseudo-embeddings for testing.        */
  /* Same text always produces same vector — enables functional tests. */

  private async getEmbedding(text: string): Promise<number[]> {
    // Try GPU embedding service first
    if (this.gpuEmbedUrl) {
      try {
        const res = await fetch(`${this.gpuEmbedUrl}/embed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, model: this.indexConfig.model }),
        });
        if (res.ok) {
          const data = (await res.json()) as any;
          return data.embedding;
        }
      } catch {
        // Fall through to CPU
      }
    }

    // CPU fallback: deterministic pseudo-embedding from text hash
    return this.hashEmbedding(text, this.indexConfig.dimension);
  }

  private hashEmbedding(text: string, dimension: number): number[] {
    const embedding = new Array(dimension);
    // Deterministic seed from text
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
    }

    // Generate pseudo-random but deterministic values using simple LCG
    let state = Math.abs(hash) || 1;
    for (let i = 0; i < dimension; i++) {
      state = (state * 1664525 + 1013904223) & 0x7fffffff;
      embedding[i] = (state / 0x7fffffff) * 2 - 1; // Range [-1, 1]
    }

    // L2 normalise for cosine similarity
    const norm = Math.sqrt(embedding.reduce((sum: number, v: number) => sum + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < dimension; i++) {
        embedding[i] = embedding[i] / norm;
      }
    }

    return embedding;
  }

  /* ── Cosine Similarity ─────────────────────────────────────────── */

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  /* ── Metadata Filtering ────────────────────────────────────────── */

  private applyFilters(docs: VectorDocument[], filters: NonNullable<VectorSearchRequest["filters"]>): VectorDocument[] {
    return docs.filter(doc => {
      const m = doc.metadata;
      if (filters.category && m.category !== filters.category) return false;
      if (filters.source && m.source !== filters.source) return false;
      if (filters.rail && !m.affectedRails.includes(filters.rail)) return false;
      if (filters.operatorClass && !m.affectedClasses.includes(filters.operatorClass)) return false;
      if (filters.activeOnly && !m.active) return false;
      if (filters.tags && filters.tags.length > 0) {
        if (!filters.tags.some(t => m.tags.includes(t))) return false;
      }
      if (filters.minConfidence) {
        const order: ConfidenceLevel[] = ["ANECDOTE", "HYPOTHESIS", "PROBABLE", "ACTIONABLE", "DOCTRINE"];
        const minIdx = order.indexOf(filters.minConfidence);
        const docIdx = order.indexOf(m.confidenceLevel);
        if (docIdx < minIdx) return false;
      }
      return true;
    });
  }

  /* ── Helpers ───────────────────────────────────────────────────── */

  private estimateTokens(text: string): number {
    return Math.ceil(text.split(/\s+/).length * 1.3);
  }

  private enforceLimit(): void {
    while (this.documents.size > this.maxDocuments) {
      const oldest = this.documents.keys().next().value!;
      this.documents.delete(oldest);
    }
  }

  removeEntry(entryId: string): number {
    let removed = 0;
    for (const [docId, doc] of this.documents) {
      if (doc.entryId === entryId) {
        this.documents.delete(docId);
        removed++;
      }
    }
    return removed;
  }

  /* ── State ─────────────────────────────────────────────────────── */

  getStats(): VectorIndexStats {
    const byCategory: Record<string, number> = {};
    const byConfidence: Record<string, number> = {};
    const bySource: Record<string, number> = {};
    const entryIds = new Set<string>();

    for (const [, doc] of this.documents) {
      entryIds.add(doc.entryId);
      byCategory[doc.metadata.category] = (byCategory[doc.metadata.category] || 0) + 1;
      byConfidence[doc.metadata.confidenceLevel] = (byConfidence[doc.metadata.confidenceLevel] || 0) + 1;
      bySource[doc.metadata.source] = (bySource[doc.metadata.source] || 0) + 1;
    }

    const avgSearch = this.searchTimes.length > 0
      ? this.searchTimes.reduce((a, b) => a + b, 0) / this.searchTimes.length
      : 0;

    return {
      totalDocuments: this.documents.size,
      totalEntries: entryIds.size,
      dimension: this.indexConfig.dimension,
      model: this.indexConfig.model,
      indexType: this.indexConfig.indexType,
      distanceMetric: this.indexConfig.distanceMetric,
      byCategory: byCategory as Record<IntelCategory, number>,
      byConfidence: byConfidence as Record<ConfidenceLevel, number>,
      bySource: bySource as Record<IntelSource, number>,
      avgChunksPerEntry: entryIds.size > 0 ? parseFloat((this.documents.size / entryIds.size).toFixed(1)) : 0,
      lastEmbeddedAt: this.lastEmbeddedAt,
      lastSearchAt: this.lastSearchAt,
      totalSearches: this.totalSearches,
      avgSearchTimeMs: parseFloat(avgSearch.toFixed(1)),
      indexSizeBytes: this.documents.size * this.indexConfig.dimension * 4,
    };
  }

  getState(): VectorServiceState {
    return {
      indexConfig: this.indexConfig,
      chunkConfig: this.chunkConfig,
      ragConfig: this.ragConfig,
      stats: this.getStats(),
      gpuAvailable: this.gpuEmbedUrl !== null,
      mode: this.gpuEmbedUrl ? "GPU_ACCELERATED" : "CPU_FALLBACK",
      whiteboardUrl: this.whiteboardUrl,
      uptime: Date.now() - this.startedAt,
    };
  }

  getIndexConfig(): VectorIndexConfig { return { ...this.indexConfig }; }
  getChunkConfig(): ChunkConfig { return { ...this.chunkConfig }; }
  getRagConfig(): RagConfig { return { ...this.ragConfig }; }
}
