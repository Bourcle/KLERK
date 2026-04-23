from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


LegalSourceType = Literal["law", "precedent", "constitutional"]
LegalTopic = Literal[
    "constitution",
    "civil",
    "criminal",
    "commercial",
    "civil_procedure",
    "criminal_procedure",
    "general",
]


class RouteDecision(BaseModel):
    source_type: LegalSourceType = "law"
    topic: LegalTopic = "general"
    collection: str = "korean_law"
    reason: str = ""


class RetrievedChunk(BaseModel):
    content: str
    similarity: float = 0.0
    source: str = "vector_db"
    title: str | None = None
    source_id: str | None = None
    collection: str | None = None
    metadata: dict = Field(default_factory=dict)


class MemoryItem(BaseModel):
    memory_id: str
    user_id: str
    topic: str
    domain: str
    content: str
    importance: float
    created_at: datetime
    last_accessed_at: datetime
    access_count: int = 0
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class MCPFetchResult(BaseModel):
    source_type: LegalSourceType
    title: str | None = None
    raw_id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class SufficiencyDecision(BaseModel):
    sufficient: bool = False
    reason: str = ""
    suggested_action: str = ""


class AnswerResult(BaseModel):
    answer: str
    route: RouteDecision
    memories: list[MemoryItem] = Field(default_factory=list)
    retrieved_docs: list[RetrievedChunk] = Field(default_factory=list)
    used_mcp: bool = False
    trace_id: str
    rewritten_query: str = ""
    retrieval_iterations: int = 0
    fallback_history: list[str] = Field(default_factory=list)
