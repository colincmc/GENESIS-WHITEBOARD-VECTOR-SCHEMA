/* ── GENESIS-WHITEBOARD-VECTOR-SCHEMA — Types ────────────────────── */
/* Vector embedding schema for Whiteboard intelligence entries.        */
/* Defines dimension, distance metric, index config, chunking.        */
/* When NeMo Retriever arrives, Whiteboard becomes a RAG vector DB.   */

/* ── Whiteboard Types (mirror — never diverge) ───────────────────── */

export type ConfidenceLevel = "ANECDOTE" | "HYPOTHESIS" | "PROBABLE" | "ACTIONABLE" | "DOCTRINE";
export type IntelCategory = "PATTERN" | "LESSON" | "CORRELATION" | "WARNING" | "ADVERSARY_INTEL" | "FAILURE_ANALYSIS";
export type IntelSource = "IRON_HALO" | "GTC" | "BRIGHTON" | "RED_TEAM" | "ACADEMY_SIM" | "MANUAL";
export type RailId = "RAIL_1" | "BEACHHEAD" | "ALL";
export type OperatorClass = "SLEEPER" | "RECON" | "CHAOS" | "DEEP_COVER" | "PAYLOAD" | "NEMO_V" | "NEMO_S" | "NEMO_X";

/* ── Embedding Configuration ─────────────────────────────────────── */

/** Distance metric for similarity search */
export type DistanceMetric = "COSINE" | "L2" | "DOT_PRODUCT";

/** Index type for approximate nearest neighbour search */
export type IndexType = "FLAT" | "IVF_FLAT" | "IVF_PQ" | "HNSW";

/** Embedding model identifier */
export type EmbeddingModel =
  | "NEMOTRON_MINI"       // NeMo 4B — fast, good for operational text
  | "E5_LARGE"            // Multilingual, 1024-dim
  | "BGE_LARGE"           // BAAI, 1024-dim
  | "ALL_MINILM"          // Sentence-transformers, 384-dim (CPU fallback)
  | "CUSTOM";             // User-provided model

/** Vector index configuration */
export interface VectorIndexConfig {
  dimension: number;               // Embedding dimension (768 default for NeMo)
  distanceMetric: DistanceMetric;  // COSINE for normalised embeddings
  indexType: IndexType;            // HNSW for <100K entries, IVF_PQ for >100K
  model: EmbeddingModel;
  // HNSW parameters
  hnswM: number;                   // Max connections per node (default 16)
  hnswEfConstruction: number;      // Build-time search depth (default 200)
  hnswEfSearch: number;            // Query-time search depth (default 100)
  // IVF parameters
  ivfNlist: number;                // Number of clusters (default 100)
  ivfNprobe: number;               // Clusters to search at query time (default 10)
  // PQ parameters
  pqM: number;                     // Sub-quantiser count (default 8)
  pqBits: number;                  // Bits per sub-quantiser (default 8)
}

/* ── Chunking Strategy ───────────────────────────────────────────── */
/* How Whiteboard entries are split for embedding.                     */
/* Intel entries are short (1-3 sentences) — mostly single-chunk.     */
/* Evidence arrays may need multi-chunk for long debrief chains.      */

export type ChunkStrategy = "SINGLE" | "SENTENCE" | "SLIDING_WINDOW" | "SEMANTIC";

export interface ChunkConfig {
  strategy: ChunkStrategy;
  maxTokens: number;               // Max tokens per chunk (default 256)
  overlapTokens: number;           // Overlap between sliding windows (default 32)
  minTokens: number;               // Minimum chunk size (default 16)
  includeMetadata: boolean;        // Prepend category/source to chunk text
  metadataTemplate: string;        // "[{category}] [{source}] [{confidence}]: "
}

/* ── Vector Document (What Gets Embedded) ────────────────────────── */

export interface VectorDocument {
  documentId: string;
  entryId: string;                 // Whiteboard IntelEntry.entryId
  chunkIndex: number;              // 0 for single-chunk, 0..N for multi-chunk
  totalChunks: number;
  text: string;                    // The text that was embedded
  embedding: number[];             // The vector (dimension matches config)
  // Metadata for filtered search
  metadata: VectorMetadata;
  embeddedAt: string;
  model: EmbeddingModel;
}

export interface VectorMetadata {
  category: IntelCategory;
  source: IntelSource;
  confidenceLevel: ConfidenceLevel;
  confidence: number;              // 0.0-1.0
  affectedRails: RailId[];
  affectedClasses: OperatorClass[];
  tags: string[];
  firstSeen: string;
  lastSeen: string;
  observations: number;
  active: boolean;
}

/* ── Search Request / Response ───────────────────────────────────── */

export interface VectorSearchRequest {
  query: string;                   // Natural language query
  queryEmbedding?: number[];       // Pre-computed embedding (skip model call)
  topK: number;                    // Number of results (default 10)
  // Metadata filters (pre-filter before vector search)
  filters?: {
    category?: IntelCategory;
    source?: IntelSource;
    minConfidence?: ConfidenceLevel;
    rail?: RailId;
    operatorClass?: OperatorClass;
    tags?: string[];
    activeOnly?: boolean;
  };
  // Search tuning
  minScore?: number;               // Minimum similarity score (0-1, default 0.5)
  rerank?: boolean;                // Apply cross-encoder reranking (default false)
  includeEmbedding?: boolean;      // Return embedding vectors in results (default false)
}

export interface VectorSearchResult {
  searchId: string;
  query: string;
  results: VectorMatch[];
  totalFound: number;
  searchTimeMs: number;
  model: EmbeddingModel;
  reranked: boolean;
}

export interface VectorMatch {
  documentId: string;
  entryId: string;
  chunkIndex: number;
  text: string;
  score: number;                   // Similarity score (0-1 for cosine)
  metadata: VectorMetadata;
  embedding?: number[];            // Only if includeEmbedding=true
}

/* ── RAG Context (What Gets Sent to LLM) ─────────────────────────── */

export interface RagContext {
  contextId: string;
  query: string;
  retrievedDocuments: VectorMatch[];
  formattedContext: string;        // Assembled prompt context
  tokenCount: number;
  maxTokens: number;
  truncated: boolean;
  assembledAt: string;
}

export interface RagConfig {
  maxContextTokens: number;        // Max tokens for retrieved context (default 2048)
  contextTemplate: string;         // How to format each result in context
  systemPrompt: string;            // System prompt for RAG queries
  includeMetadata: boolean;        // Include category/confidence in context
  deduplicateEntries: boolean;     // Merge chunks from same entry
  boostDoctrine: boolean;          // Weight DOCTRINE entries higher
  doctrineBoostFactor: number;     // Multiplier for DOCTRINE scores (default 1.5)
}

/* ── Embedding Pipeline ──────────────────────────────────────────── */

/** Request to embed a Whiteboard entry */
export interface EmbedRequest {
  entryId: string;
  intelligence: string;            // The text to embed
  evidence?: string[];             // Additional evidence text
  category: IntelCategory;
  source: IntelSource;
  confidenceLevel: ConfidenceLevel;
  confidence: number;
  affectedRails: RailId[];
  affectedClasses: OperatorClass[];
  tags: string[];
  firstSeen: string;
  lastSeen: string;
  observations: number;
}

/** Batch embed request */
export interface BatchEmbedRequest {
  entries: EmbedRequest[];
}

/** Embed response */
export interface EmbedResponse {
  entryId: string;
  documents: VectorDocument[];
  chunksCreated: number;
  embeddingTimeMs: number;
}

/* ── Index Statistics ────────────────────────────────────────────── */

export interface VectorIndexStats {
  totalDocuments: number;
  totalEntries: number;            // Unique Whiteboard entries embedded
  dimension: number;
  model: EmbeddingModel;
  indexType: IndexType;
  distanceMetric: DistanceMetric;
  byCategory: Record<IntelCategory, number>;
  byConfidence: Record<ConfidenceLevel, number>;
  bySource: Record<IntelSource, number>;
  avgChunksPerEntry: number;
  lastEmbeddedAt: string | null;
  lastSearchAt: string | null;
  totalSearches: number;
  avgSearchTimeMs: number;
  indexSizeBytes: number;          // Approximate in-memory size
}

/* ── Service State ───────────────────────────────────────────────── */

export interface VectorServiceState {
  indexConfig: VectorIndexConfig;
  chunkConfig: ChunkConfig;
  ragConfig: RagConfig;
  stats: VectorIndexStats;
  gpuAvailable: boolean;
  mode: "CPU_FALLBACK" | "GPU_ACCELERATED";
  whiteboardUrl: string | null;
  uptime: number;
}
