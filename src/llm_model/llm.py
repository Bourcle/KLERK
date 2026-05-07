import json
import os
from typing import Any
from urllib.parse import urljoin
from pathlib import Path

import httpx
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import Settings
from utils.exceptions import (
    ConfigError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
)


def normalize_provider_name(provider: str) -> str:
    """Normalize a provider name for consistent backend selection.

    Args:
        provider: Raw provider name from settings.

    Returns:
        str: Lowercase provider name with surrounding spaces removed and hyphens replaced by underscores.
    """

    return (provider or "").strip().lower().replace("-", "_")


class LLMFactory:
    """Factory for validating settings and creating LLM and embedding backends."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_settings(self) -> None:
        """Validate required LLM backend settings before model creation.

        Raises:
            ConfigError: If required model, base URL, or API key settings are missing.
        """

        provider = normalize_provider_name(self.settings.local_llm_provider)
        model = self.settings.local_llm_model.strip()
        if not model:
            raise ConfigError(
                "LOCAL_LLM_MODEL is empty",
                user_message="LLM 모델명이 비어 있습니다. .env의 LOCAL_LLM_MODEL 값을 확인해 주세요.",
            )
        if provider == "ollama":
            base_url = self.settings.local_llm_base_url.strip()
            if not base_url:
                raise ConfigError(
                    "LOCAL_LLM_BASE_URL is empty for ollama",
                    user_message="Ollama 서버 주소가 비어 있습니다. .env의 LOCAL_LLM_BASE_URL 값을 확인해 주세요.",
                )
        if provider in {"openai_compatible", "vllm"}:
            base_url = self.settings.local_llm_base_url.strip()
            if not base_url:
                raise ConfigError(
                    "LOCAL_LLM_BASE_URL is empty for OpenAI-compatible backend",
                    user_message="OpenAI 호환 서버 주소가 비어 있습니다. .env의 LOCAL_LLM_BASE_URL 값을 확인해 주세요.",
                )
            if provider == "openai_compatible" and not (self.settings.local_llm_api_key or "").strip():
                raise ConfigError(
                    "LOCAL_LLM_API_KEY is empty for openai_compatible",
                    user_message="OpenAI 호환 API 키가 비어 있습니다. .env의 LOCAL_LLM_API_KEY 값을 확인해 주세요.",
                )

    def create_chat_model(self):
        """Create a chat model instance from the configured local LLM provider.

        Returns:
            ChatOllama | ChatOpenAI: Chat model instance configured from runtime settings.

        Raises:
            ConfigError: If the configured LLM provider is not supported.
        """

        provider = normalize_provider_name(self.settings.local_llm_provider)
        if provider == "ollama":
            return ChatOllama(
                model=self.settings.local_llm_model,
                base_url=self.settings.local_llm_base_url,
                temperature=self.settings.local_llm_temperature,
                num_predict=self.settings.local_llm_max_tokens,
            )
        if provider in {"openai_compatible", "vllm"}:
            return ChatOpenAI(
                model=self.settings.local_llm_model,
                base_url=self.settings.local_llm_base_url,
                api_key=self.settings.local_llm_api_key or "EMPTY",
                temperature=self.settings.local_llm_temperature,
                max_tokens=self.settings.local_llm_max_tokens,
            )
        raise ConfigError(f"Unsupported LLM provider: {self.settings.local_llm_provider}")

    def create_embeddings(self):
        """Create an embedding model instance from the configured embedding provider.

        Returns:
            OllamaEmbeddings | HuggingFaceEmbeddings | OpenAIEmbeddings: Embedding backend instance.

        Raises:
            ConfigError: If the configured embedding provider is not supported.
        """

        provider = normalize_provider_name(self.settings.local_embedding_provider)
        if provider == "ollama":
            return OllamaEmbeddings(
                model=self.settings.local_embedding_model,
                base_url=self.settings.local_embedding_base_url,
            )
        if provider in {"huggingface_local", "local"}:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            model_name = resolve_local_hf_snapshot(self.settings.local_embedding_model)
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"local_files_only": True},
            )
        if provider == "openai_compatible":
            return OpenAIEmbeddings(
                model=self.settings.local_embedding_model,
                base_url=self.settings.local_embedding_base_url,
                api_key=self.settings.local_embedding_api_key or self.settings.local_llm_api_key or "EMPTY",
            )
        raise ConfigError(f"Unsupported embedding provider: {self.settings.local_embedding_provider}")

    async def acheck_chat_backend(self) -> None:
        """Run an asynchronous health check for the configured chat backend.

        Returns:
            None: This method completes silently when the backend check succeeds.
        """

        provider = normalize_provider_name(self.settings.local_llm_provider)
        if provider == "ollama":
            await self.acheck_ollama_backend()

    async def acheck_ollama_backend(self) -> None:
        """Check whether the configured Ollama server and model are available.

        Returns:
            None: This method completes silently when Ollama and the target model are available.

        Raises:
            ConfigError: If the Ollama base URL is invalid.
            LLMConnectionError: If the Ollama server cannot be reached or returns an invalid response.
            LLMModelNotFoundError: If the configured Ollama model is not installed.
        """

        base_url = self.settings.local_llm_base_url.rstrip("/")
        model_name = self.settings.local_llm_model
        timeout = httpx.Timeout(5.0, connect=3.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(urljoin(f"{base_url}/", "api/tags"))
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {base_url}",
                user_message=(
                    "Ollama 서버에 연결할 수 없습니다. " f"{base_url}에서 Ollama가 실행 중인지 확인해 주세요."
                ),
            ) from exc
        except httpx.InvalidURL as exc:
            raise ConfigError(
                f"Invalid Ollama base URL: {base_url}",
                user_message="Ollama 서버 주소 형식이 올바르지 않습니다. .env의 LOCAL_LLM_BASE_URL 값을 확인해 주세요.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMConnectionError(
                f"Ollama health check returned {exc.response.status_code}",
                user_message="Ollama 서버 응답을 확인할 수 없습니다. 서버 상태와 주소를 확인해 주세요.",
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f"Ollama health check failed for {base_url}: {exc}",
                user_message="Ollama 서버 상태 확인 중 네트워크 오류가 발생했습니다.",
            ) from exc

        payload = response.json()
        models = payload.get("models", [])
        installed_names = {item.get("model") or item.get("name") for item in models if isinstance(item, dict)}
        if model_name not in installed_names:
            installed = ", ".join(sorted(name for name in installed_names if name)) or "(none)"
            raise LLMModelNotFoundError(
                f"Ollama model '{model_name}' not found. Installed models: {installed}",
                user_message=(
                    f"Ollama 서버에는 {model_name} 모델이 없습니다. "
                    f"ollama pull {model_name}로 모델을 준비해 주세요."
                ),
            )


def raise_for_http_error(exc: httpx.HTTPStatusError, *, provider: str, base_url: str, model_name: str) -> None:
    """Convert an HTTP status error into a domain-specific LLM exception.

    Args:
        exc: HTTP status error raised during model server communication.
        provider: Name of the model provider or client class.
        base_url: Base URL of the model server.
        model_name: Requested model name.

    Raises:
        LLMAuthenticationError: If authentication or authorization fails.
        LLMModelNotFoundError: If the requested model is not found.
        LLMError: If the model server request fails for another HTTP status.
    """

    status = exc.response.status_code
    if status in {401, 403}:
        raise LLMAuthenticationError(
            f"{provider} authentication failed with status {status}",
            user_message="모델 서버 인증에 실패했습니다. API 키 또는 권한 설정을 확인해 주세요.",
        ) from exc
    if status == 404:
        raise LLMModelNotFoundError(
            f"{provider} model '{model_name}' not found at {base_url}",
            user_message=(
                f"모델 서버에서 {model_name} 모델을 찾지 못했습니다. " "모델 이름과 서버 endpoint를 확인해 주세요."
            ),
        ) from exc
    raise LLMError(
        f"{provider} request failed with status {status}",
        user_message="모델 서버 요청이 실패했습니다. 서버 상태와 설정을 확인해 주세요.",
    ) from exc


def resolve_local_hf_snapshot(model_name: str) -> str:
    """Resolve a locally cached Hugging Face snapshot path when available.

    Args:
        model_name: Hugging Face model name or local model path.

    Returns:
        str: Local snapshot path when a valid cached snapshot exists; otherwise the original model name.
    """

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name
    candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)
    required_files = ("config.json", "modules.json", "tokenizer.json")
    for snapshot in candidates:
        if all((snapshot / name).exists() for name in required_files):
            return str(snapshot)
    return model_name


async def ainvoke_text(model: Any, messages: list[dict[str, str]]) -> str:
    """Invoke a chat model asynchronously and return its response as plain text.

    Args:
        model: Chat model object exposing an asynchronous ainvoke method.
        messages: Chat messages to send to the model.

    Returns:
        str: Model response content converted to text.

    Raises:
        LLMConnectionError: If the model server cannot be reached.
        LLMAuthenticationError: If model server authentication fails.
        LLMModelNotFoundError: If the requested model is not found.
        LLMError: If model invocation fails for another reason.
    """

    try:
        response = await model.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)
    except httpx.ConnectError as exc:
        model_name = getattr(model, "model", "unknown")
        base_url = getattr(model, "base_url", None) or getattr(model, "_base_url", None) or "unknown"
        provider = model.__class__.__name__
        provider_hint = "Ollama" if "Ollama" in provider else "모델 서버"
        raise LLMConnectionError(
            f"{provider} connection failed to {base_url}",
            user_message=(
                f"{provider_hint}에 연결할 수 없습니다. "
                f"서버 주소({base_url})와 실행 상태, 모델 설정({model_name})을 확인해 주세요."
            ),
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise_for_http_error(
            exc,
            provider=model.__class__.__name__,
            base_url=str(getattr(model, "base_url", "unknown")),
            model_name=str(getattr(model, "model", "unknown")),
        )
    except Exception as exc:
        raise LLMError(
            str(exc),
            user_message="질문 처리 중 모델 호출에 실패했습니다. 서버 설정과 로그를 확인해 주세요.",
        ) from exc


async def ainvoke_json(
    model: Any,
    messages: list[dict[str, str]],
    default: dict[str, Any],
) -> dict[str, Any]:
    """Invoke a chat model asynchronously and parse the response as JSON.

    Args:
        model: Chat model object exposing an asynchronous ainvoke method.
        messages: Chat messages to send to the model.
        default: Fallback dictionary returned when JSON parsing fails.

    Returns:
        dict[str, Any]: Parsed JSON response, or the provided default value when parsing fails.
    """

    raw = await ainvoke_text(model, messages)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return default
        return json.loads(raw[start : end + 1])
    except Exception:
        return default
