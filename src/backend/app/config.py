# src/backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App-level config only (no secrets here)
    API_TITLE: str = "Text2SQL API"
    API_VERSION: str = "1.0.0"
    JSON_ROW_LIMIT: int = 200
    CSV_DEFAULT_LIMIT: int | None = None
    ALLOW_ORIGINS: list[str] = ["*"]

    # Pydantic v2 settings
    model_config = SettingsConfigDict(
        env_file=".env",            
        env_file_encoding="utf-8",
        extra="ignore",             
        case_sensitive=False,
    )

settings = Settings()