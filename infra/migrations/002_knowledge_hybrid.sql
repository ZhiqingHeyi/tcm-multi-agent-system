-- ============================================================================
-- 知识库检索层升级（幂等，可重复执行）
--
-- 三项改进：
-- 1. 补齐检索所需元数据（title / role / chunk_index / bigram 索引列）
-- 2. 稠密侧：ivfflat → HNSW。HNSW 无需预先聚类训练、增量插入友好、召回更稳，
--    是生产环境 pgvector 的默认选择（m=16, ef_construction=64 为通用起点）。
-- 3. 稀疏侧：中文 bigram 落库 + GIN 索引，把"候选生成"下推到数据库。
--    原来在 Python 里全表扫描算 BM25，10 万块级别就会拖垮接口；
--    现在由索引先召回数百候选，再在应用层精确打分（召回靠索引，精度靠精确算分）。
-- ============================================================================

ALTER TABLE knowledge_chunks
    ADD COLUMN IF NOT EXISTS title       VARCHAR(200) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS role        VARCHAR(20)  NOT NULL DEFAULT 'classic',
    ADD COLUMN IF NOT EXISTS chunk_index INTEGER      NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS content_bigrams TEXT    NOT NULL DEFAULT '';

-- 稀疏召回索引：bigram 以空格分隔落库，用 simple 配置做 tsvector（不依赖 zhparser）
CREATE INDEX IF NOT EXISTS knowledge_chunks_bigrams_idx
    ON knowledge_chunks USING gin (to_tsvector('simple', content_bigrams));

-- 稠密召回索引
DROP INDEX IF EXISTS knowledge_chunks_embedding_idx;
CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_hnsw_idx
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 组合过滤：按学派 + 文档角色预筛，避免学派间知识串味
CREATE INDEX IF NOT EXISTS knowledge_chunks_school_role_idx
    ON knowledge_chunks (school, role);

CREATE INDEX IF NOT EXISTS knowledge_chunks_source_idx
    ON knowledge_chunks (source);
