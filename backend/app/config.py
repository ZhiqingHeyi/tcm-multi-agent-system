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
    admin_token: str = "tcm-admin"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
BACKEND_DIR = Path(__file__).resolve().parents[1]
