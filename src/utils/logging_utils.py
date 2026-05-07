import logging
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import orjson


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in [
            "trace_id",
            "thread_id",
            "user_id",
            "node",
            "event",
            "duration_ms",
            "error_type",
            "route",
            "collection",
            "provider",
            "model",
            "base_url",
            "embedding_provider",
            "embedding_model",
            "rewritten_query",
            "retrieval_count",
            "top_score",
            "rerank_scores",
            "sufficiency_decision",
            "sufficiency_reason",
            "fallback_action",
            "fallback_iteration",
            "mcp_called",
            "mcp_upserted",
            "selected_collection",
            "source_type",
            "topic",
            "recovery_steps",
            "citation_valid",
        ]:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return orjson.dumps(payload).decode("utf-8")


class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _configured = True


def get_logger(name: str, **context: Any) -> ContextLoggerAdapter:
    return ContextLoggerAdapter(logging.getLogger(name), context)


def log_event(logger: ContextLoggerAdapter, event_name: str, **payload: Any) -> None:
    logger.info(event_name, extra={"event": event_name, **payload})


def new_trace_id() -> str:
    return uuid4().hex
