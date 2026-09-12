from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://tcm:tcm@localhost:5432/tcm"
    jwt_secret: str = "change-this-in-production"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model_fast: str = "deepseek-v4-flash-0731"
    llm_model_pro: str = "deepseek-v4-pro-0813"
    vision_api_key: str = ""
    vision_model: str = "gpt-4o-mini"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1024
    rag_top_k: int = 4
    rag_rerank: bool = True
    # 稀疏通道在 RRF 融合中的权重。评测显示稀疏通道单独召回率显著低于稠密通道，
    # 等权融合会把它的噪声排名带进最终结果（严格标签下融合 37.5% < 稠密 50%），
    # 因此按通道可信度降权，而非等权。
    rag_sparse_weight: float = 0.4
    # 分层配额检索：共享基础层（common）在 Top-K 中固定预留的席位。
    # 不设配额时，占比三分之一的泛中医典籍会挤占本派权威典籍的位置。
    rag_tiered: bool = True
    rag_common_quota: int = 1
    admin_token: str = "tcm-admin"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
BACKEND_DIR = Path(__file__).resolve().parents[1]
