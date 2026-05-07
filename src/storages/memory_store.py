import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from parsers import overlap_ratio, stable_id, tokenize_koreanish
from data_structure.schemas import MemoryItem, RouteDecision


class MemoryRepository:
    """SQLite-backed repository for storing, searching, and updating user memories."""

    def __init__(self, db_path: str, half_life_days: int = 30, default_importance: float = 0.55):
        self.db_path = db_path
        self.half_life_days = half_life_days
        self.default_importance = default_importance
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        """Create a SQLite connection configured to return rows by column name.

        Returns:
            sqlite3.Connection: SQLite connection with row factory set to sqlite3.Row.
        """

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create the memories table if it does not already exist.

        Returns:
            None: This method completes after the database schema is initialized.
        """

        with closing(self.connect()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """)
            conn.commit()

    def add_memory(
        self,
        *,
        user_id: str,
        topic: str,
        domain: str,
        content: str,
        importance: float | None = None,
        metadata_json: str = "{}",
    ) -> str:
        """Insert or replace a memory record and return its stable ID.

        Args:
            user_id: User identifier associated with the memory.
            topic: Topic label used to group or describe the memory.
            domain: Domain label used for domain-aware retrieval.
            content: Memory content to store.
            importance: Optional importance score overriding the repository default.
            metadata_json: JSON string containing additional memory metadata.

        Returns:
            str: Stable memory ID generated from user, topic, domain, and content.
        """

        now = datetime.now(timezone.utc).isoformat()
        memory_id = stable_id(user_id, topic, domain, content[:120])
        with closing(self.connect()) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    memory_id, user_id, topic, domain, content, importance,
                    created_at, last_accessed_at, access_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((
                    SELECT access_count FROM memories WHERE memory_id = ?
                ), 0), ?)
                """,
                (
                    memory_id,
                    user_id,
                    topic,
                    domain,
                    content,
                    float(importance if importance is not None else self.default_importance),
                    now,
                    now,
                    memory_id,
                    metadata_json,
                ),
            )
            conn.commit()
        return memory_id

    def freshness(self, created_at: datetime) -> float:
        """Calculate an exponential freshness score from memory age.

        Args:
            created_at: UTC datetime when the memory was created.

        Returns:
            float: Freshness score decayed by the configured half-life.
        """

        age_days = max((datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0, 0.0)
        return math.exp(-math.log(2) * age_days / max(self.half_life_days, 1))

    def search(self, *, user_id: str, query: str, domain: str, top_k: int = 3) -> list[MemoryItem]:
        """Search and rank user memories by relevance, freshness, importance, and domain match.

        Args:
            user_id: User identifier used to filter memories.
            query: Current user question or search text.
            domain: Domain label used to boost matching memories.
            top_k: Maximum number of memories to return.

        Returns:
            list[MemoryItem]: Ranked memory items with computed relevance scores.
        """

        query_tokens = tokenize_koreanish(query)
        results: list[MemoryItem] = []

        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

        for row in rows:
            created_at = datetime.fromisoformat(row["created_at"])
            last_accessed_at = datetime.fromisoformat(row["last_accessed_at"])
            topic_tokens = tokenize_koreanish(row["topic"] + " " + row["content"])
            topic_overlap = overlap_ratio(query_tokens, topic_tokens)
            freshness = self.freshness(created_at)
            domain_boost = 1.15 if row["domain"] == domain else 1.0
            access_boost = 1.0 + min(row["access_count"], 10) * 0.03
            score = float(row["importance"]) * freshness * (0.5 + topic_overlap) * domain_boost * access_boost
            results.append(
                MemoryItem(
                    memory_id=row["memory_id"],
                    user_id=row["user_id"],
                    topic=row["topic"],
                    domain=row["domain"],
                    content=row["content"],
                    importance=float(row["importance"]),
                    created_at=created_at,
                    last_accessed_at=last_accessed_at,
                    access_count=int(row["access_count"]),
                    score=score,
                    metadata={},
                )
            )

        ranked = sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
        self.touch_many([item.memory_id for item in ranked])
        return ranked

    def touch_many(self, memory_ids: list[str]) -> None:
        """Update access metadata for multiple memories.

        Args:
            memory_ids: Memory IDs to mark as accessed.

        Returns:
            None: This method completes after updating access counts and timestamps.
        """

        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        with closing(self.connect()) as conn:
            conn.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE memory_id = ?",
                [(now, memory_id) for memory_id in memory_ids],
            )
            conn.commit()

    def record_turn(self, *, user_id: str, question: str, route: RouteDecision, answer: str) -> None:
        """Store a compact summary of a completed question-answer turn as memory.

        Args:
            user_id: User identifier associated with the turn.
            question: Normalized user question.
            route: Route decision containing topic and domain information.
            answer: Generated answer text to summarize and store.

        Returns:
            None: This method completes after storing the turn summary or skipping empty answers.
        """

        compact_answer = answer[:400]
        if not compact_answer.strip():
            return
        topic = route.topic
        domain = route.source_type
        importance = (
            0.7
            if any(token in question for token in ["판례", "제", "조", "손해배상", "형사", "민법", "형법"])
            else self.default_importance
        )
        content = f"Q: {question}\nA: {compact_answer}"
        self.add_memory(
            user_id=user_id,
            topic=topic,
            domain=domain,
            content=content,
            importance=importance,
            metadata_json='{"kind": "turn_summary"}',
        )
