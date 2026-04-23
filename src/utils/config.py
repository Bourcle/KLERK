import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    local_llm_provider: str = Field(default="ollama", alias="LOCAL_LLM_PROVIDER")
    local_llm_model: str = Field(default="qwen3:8b", alias="LOCAL_LLM_MODEL")
    local_llm_base_url: str = Field(default="http://localhost:11434", alias="LOCAL_LLM_BASE_URL")
    local_llm_api_key: str | None = Field(default=None, alias="LOCAL_LLM_API_KEY")
    local_llm_temperature: float = Field(default=0.1, alias="LOCAL_LLM_TEMPERATURE")
    local_llm_max_tokens: int = Field(default=2048, alias="LOCAL_LLM_MAX_TOKENS")

    local_embedding_provider: str = Field(default="ollama", alias="LOCAL_EMBEDDING_PROVIDER")
    local_embedding_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("LOCAL_EMB_MODEL", "LOCAL_EMBEDDING_MODEL"),
    )
    local_embedding_base_url: str = Field(default="http://localhost:11434", alias="LOCAL_EMBEDDING_BASE_URL")
    local_embedding_api_key: str | None = Field(default=None, alias="LOCAL_EMBEDDING_API_KEY")

    checkpoint_sqlite_path: str = Field(default=".data/checkpoints.sqlite", alias="CHECKPOINT_SQLITE_PATH")
    memory_sqlite_path: str = Field(default=".data/memory.sqlite", alias="MEMORY_SQLITE_PATH")
    chroma_persist_dir: str = Field(default=".data/chroma", alias="CHROMA_PERSIST_DIR")

    default_collection: str = Field(default="korean_law", alias="DEFAULT_COLLECTION")
    use_topic_collections: bool = Field(default=True, alias="USE_TOPIC_COLLECTIONS")
    vector_top_k: int = Field(default=4, alias="VECTOR_TOP_K")
    mcp_fetch_top_n: int = Field(default=3, alias="MCP_FETCH_TOP_N")
    similarity_threshold: float = Field(default=0.40, alias="SIMILARITY_THRESHOLD")
    min_retrieved_docs: int = Field(default=2, alias="MIN_RETRIEVED_DOCS")
    max_context_chars: int = Field(default=5000, alias="MAX_CONTEXT_CHARS")
    max_retrieval_iterations: int = Field(default=3, alias="MAX_RETRIEVAL_ITERATIONS")
    rerank_top_k: int = Field(default=4, alias="RERANK_TOP_K")

    memory_half_life_days: int = Field(default=30, alias="MEMORY_HALF_LIFE_DAYS")
    memory_default_importance: float = Field(default=0.55, alias="MEMORY_DEFAULT_IMPORTANCE")
    memory_top_k: int = Field(default=3, alias="MEMORY_TOP_K")

    gradio_host: str = Field(default="127.0.0.1", alias="GRADIO_HOST")
    gradio_port: int = Field(default=7860, alias="GRADIO_PORT")
    char_stream_delay: float = Field(default=0.01, alias="CHAR_STREAM_DELAY")

    law_api_oc: str | None = Field(default=None, alias="LAW_API_OC")
    mcp_server_command: str = Field(default="node", alias="MCP_SERVER_COMMAND")
    mcp_server_entrypoint: str = Field(default="./external/korean-law-mcp/dist/index.js", alias="MCP_SERVER_ENTRYPOINT")

    @property
    def checkpoint_path(self) -> str:
        return self.checkpoint_sqlite_path

    @property
    def memory_path(self) -> str:
        return self.memory_sqlite_path

    @property
    def chroma_path(self) -> str:
        return self.chroma_persist_dir

    def ensure_directories(self) -> None:
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        memory_dir = os.path.dirname(self.memory_path)
        chroma_dir = self.chroma_path

        for path in [checkpoint_dir, memory_dir, chroma_dir]:
            if path:
                os.makedirs(path, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
