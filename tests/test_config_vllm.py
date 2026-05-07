from pathlib import Path

from utils.config import Settings


def test_settings_loads_vllm_provider_from_env_file(tmp_path: Path, monkeypatch):
    for key in [
        "LOCAL_LLM_PROVIDER",
        "LOCAL_LLM_BASE_URL",
        "LOCAL_LLM_MODEL",
        "LOCAL_LLM_API_KEY",
        "LOCAL_EMBEDDING_PROVIDER",
        "LOCAL_EMBEDDING_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "LOCAL_LLM_PROVIDER=vllm",
                "LOCAL_LLM_BASE_URL=http://localhost:8000/v1",
                "LOCAL_LLM_MODEL=test-local-model",
                "LOCAL_LLM_API_KEY=dummy",
                "LOCAL_EMBEDDING_PROVIDER=local",
                "LOCAL_EMBEDDING_MODEL=BAAI/bge-m3",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_path)

    assert settings.local_llm_provider == "vllm"
    assert settings.local_llm_base_url == "http://localhost:8000/v1"
    assert settings.local_llm_model == "test-local-model"
    assert settings.local_embedding_provider == "local"
    assert settings.local_embedding_model == "BAAI/bge-m3"
