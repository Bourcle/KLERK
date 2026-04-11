class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str = "", *, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or "요청을 처리하는 중 오류가 발생했습니다."


class ConfigError(AppError):
    """Raised when configuration is invalid or incomplete."""


class LLMError(AppError):
    """Raised when an LLM request fails."""


class LLMConnectionError(LLMError):
    """Raised when the LLM backend cannot be reached."""


class LLMModelNotFoundError(LLMError):
    """Raised when the configured model is not available on the backend."""


class LLMAuthenticationError(LLMError):
    """Raised when the backend rejects credentials."""


class RoutingError(AppError):
    """Raised when question routing fails."""


class VectorStoreError(AppError):
    """Raised when vector DB operations fail."""


class MCPError(AppError):
    """Raised when MCP server interaction fails."""


class GenerationError(AppError):
    """Raised when final answer generation fails."""
