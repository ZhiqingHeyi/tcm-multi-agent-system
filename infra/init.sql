CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    school VARCHAR(40) NOT NULL,
    source VARCHAR(200) NOT NULL,
    section VARCHAR(300) NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    token_key VARCHAR(640) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx
    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS knowledge_chunks_school_idx
    ON knowledge_chunks (school);
