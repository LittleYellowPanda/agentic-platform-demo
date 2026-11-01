"""
Pydantic way to read and validate the environment variables from the .env file.
Enables display of clear error messages when the environment variables are not set or invalid.

"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "db"
    postgres_port: int = 5432

    cors_origins: str = "http://localhost:5173"

    # Where embedding model weights are stored. None = the library's default folder.
    # In Docker, points to a volume so models aren't re-downloaded at each rebuild.
    model_cache_dir: str | None = None

    # Uploads larger than this are rejected (413) before being stored or indexed
    max_upload_size_mb: int = 20

    # LLM: any OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, a company gateway).
    # "openrouter/free" lets OpenRouter pick a free model that supports tool calling.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = (
        ""  # empty = mock mode: the app runs with rule-based fake answers
    )
    llm_model: str = "openrouter/free"
    # Safety net: an agent stops after this many LLM calls, answered or not
    agent_max_iterations: int = 6
    # Previous turns (question + final answer) resent with each new question
    conversation_max_turns: int = 10
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2  # 429, 5xx, connection errors; backoff handled by the SDK
    tavily_api_key: str = ""  # web search tool; disabled when empty
    # Extra providers an agent can pick, besides the default one above (optional).
    # Ollama: e.g. http://host.docker.internal:11434/v1 (no key needed)
    ollama_base_url: str = ""
    openrouter_api_key: str = ""

    @property  # turns a method into a property that can be accessed like an attribute
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Create an instance of the Settings class to be used throughout the application
settings = Settings()
