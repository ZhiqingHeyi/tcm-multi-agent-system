-- 全量建库脚本（容器首次启动时由 docker-entrypoint-initdb.d 执行）。
-- 与 infra/migrations/ 下的迁移文件保持一致，二者必须同步修改。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id              BIGSERIAL PRIMARY KEY,
    school          VARCHAR(40)  NOT NULL,
    source          VARCHAR(200) NOT NULL,
    section         VARCHAR(300) NOT NULL DEFAULT '',
    title           VARCHAR(200) NOT NULL DEFAULT '',
    role            VARCHAR(20)  NOT NULL DEFAULT 'classic',
    chunk_index     INTEGER      NOT NULL DEFAULT 0,
    content         TEXT         NOT NULL,
    content_bigrams TEXT         NOT NULL DEFAULT '',
    embedding       VECTOR(1024) NOT NULL,
    token_key       VARCHAR(640) NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 稀疏召回：中文 bigram 的 GIN 索引（候选生成下推到 DB）
CREATE INDEX IF NOT EXISTS knowledge_chunks_bigrams_idx
    ON knowledge_chunks USING gin (to_tsvector('simple', content_bigrams));

-- 稠密召回：HNSW 余弦索引
CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_hnsw_idx
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS knowledge_chunks_school_role_idx ON knowledge_chunks (school, role);
CREATE INDEX IF NOT EXISTS knowledge_chunks_source_idx ON knowledge_chunks (source);
