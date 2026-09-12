-- 003: 向量化空间护栏
--
-- 换 embedding 模型等于换向量空间。若库里同时存在两个模型产生的向量，
-- 新查询向量与旧向量算余弦相似度是纯噪声，且不会报错——只会表现为
-- "召回莫名变差"，极难排查。embedder_id 让每个块记录自己的向量空间，
-- 检索时用 WHERE embedder_id = :current 硬过滤，从根上杜绝混用。
--
-- 注意：本迁移执行后，存量块的 embedder_id 为空字符串，向量检索会查不到，
-- 直到用新模型重嵌入（python -m app.rag.ingest --force）。这是有意设计：
-- 宁可暂时查不到，也不返回错答案。

ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS embedder_id VARCHAR(80) NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedder
    ON knowledge_chunks (embedder_id);

-- 若未来更换的模型输出维度不是 1024，需要先改列再重建索引（模板如下）：
-- ALTER TABLE knowledge_chunks ALTER COLUMN embedding TYPE vector(768);
-- REINDEX INDEX knowledge_chunks_embedding_hnsw_idx;
