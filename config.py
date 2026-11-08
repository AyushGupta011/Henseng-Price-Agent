from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Secrets
    APIFY_TOKEN: str | None = None
    EXA_API_KEY: str | None = None
    LLM_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None
    LLM_MODEL: str | None = None

    def model_post_init(self, __context: object) -> None:
        # 1. Resolve API key
        if self.GROQ_API_KEY and not self.LLM_API_KEY:
            self.LLM_API_KEY = self.GROQ_API_KEY

        is_groq = (
            (self.LLM_API_KEY is not None and self.LLM_API_KEY.startswith("gsk_"))
            or bool(self.GROQ_API_KEY)
            or (self.LLM_BASE_URL is not None and "groq.com" in self.LLM_BASE_URL)
        )

        # 2. Dynamic Base URL (respects user choice if set in .env)
        if not self.LLM_BASE_URL:
            if is_groq:
                self.LLM_BASE_URL = "https://api.groq.com/openai/v1"
            else:
                self.LLM_BASE_URL = "https://api.openai.com/v1"

        # 3. Dynamic Model (strictly respects user choice if set in .env)
        if not self.LLM_MODEL:
            if is_groq:
                self.LLM_MODEL = "qwen/qwen3.8-27b"
            else:
                self.LLM_MODEL = "gpt-4o-mini"

    # Data paths
    DATA_DIR: Path = Path("data")
    RAW_CACHE_DIR: Path = Path("data/raw")
    CACHE_DIR: Path = Path("data/cache")
    REPORTS_DIR: Path = Path("data/reports")

    # Pipeline toggles
    RAW_CACHE_ENABLED: bool = True
    LLM_CACHE_ENABLED: bool = True
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.5

    def ensure_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
