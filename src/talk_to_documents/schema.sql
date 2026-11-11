CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title       text NOT NULL,
  source_type text NOT NULL DEFAULT 'upload'
              CHECK (source_type IN ('upload','conversation','note')),
  filename    text,
  sha256      text NOT NULL UNIQUE,
  fts_config  text NOT NULL DEFAULT 'simple',
  embed_model text NOT NULL,
  tags        text[] NOT NULL DEFAULT '{}',
  metadata    jsonb NOT NULL DEFAULT '{}',
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id    uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal        int  NOT NULL,
  page           int,
  heading_path   text,
  content        text NOT NULL,
  context_prefix text,
  token_count    int  NOT NULL,
  embedding      vector(1024) NOT NULL,  -- use the real dimension from Task 4
  tsv            tsvector NOT NULL,
  UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);