from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    trace_id: str
    question: str
    normalized_question: str
    route: dict[str, Any]
    memories: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    retrieval_sufficient: bool
    used_mcp: bool
    answer: str
    error: str | None
    rewritten_query: str
    retrieval_iterations: int
    fallback_history: list[str]


class MemoryState(TypedDict, total=False):
    user_id: str
    question: str
    route: dict[str, Any]
    memories: list[dict[str, Any]]


class LegalRAGState(TypedDict, total=False):
    question: str
    route: dict[str, Any]
    memories: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    retrieval_sufficient: bool
    used_mcp: bool
    answer: str
    rewritten_query: str
    iteration: int
    fallback_history: list[str]
