"""Application configuration module."""
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    pg_host: str = Field(..., alias="PGHOST")
    pg_port: int = Field(default=5432, alias="PGPORT")
    pg_user: str = Field(..., alias="PGUSER")
    pg_password: str = Field(default="", alias="PGPASSWORD")
    pg_database: str = Field(..., alias="PGDATABASE")
    pgvector_schema: str = Field(default="public", alias="PGVECTOR_SCHEMA")

    model_dir: Path = Field(default=Path("/app/models"), alias="MODEL_DIR")
    sentence_model_path: Path = Field(default=Path("/app/models/sentence_model"), alias="SENTENCE_MODEL_PATH")

    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "populate_by_name": True,
        "protected_namespaces": ("settings_",),
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
