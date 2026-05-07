from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config import Settings
from utils.exceptions import VectorStoreError
from utils.logging_utils import get_logger
from parsers import stable_id
from data_structure.schemas import MCPFetchResult, RetrievedChunk, RouteDecision


def distance_to_similarity(distance: float | None) -> float:
    """Convert a vector distance score into a bounded similarity score.

    Args:
        distance: Raw distance value returned from vector search.

    Returns:
        float: Similarity score where smaller distances produce higher similarity.
    """

    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(float(distance), 0.0))


def open_vector_db(*, embeddings, persist_dir: Path | str, collection: str) -> Chroma:
    """Initialize and return a Chroma vector database instance.

    Args:
        embeddings: Embedding function used by the Chroma collection.
        persist_dir: Directory path where Chroma data is persisted.
        collection: Chroma collection name to open or create.

    Returns:
        Chroma: Configured Chroma vector database instance.
    """

    return Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


class LegalVectorStore:
    """Vector store wrapper for legal document retrieval and MCP result caching."""

    def __init__(self, persist_dir: Path, embeddings, settings: Settings):
        self.persist_dir = Path(persist_dir)
        self.embeddings = embeddings
        self.settings = settings
        self._stores: dict[str, Chroma] = {}
        self.logger = get_logger(__name__)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def get_store(self, collection_name: str) -> Chroma:
        """Get or lazily initialize a Chroma collection by name.

        Args:
            collection_name: Name of the Chroma collection to load.

        Returns:
            Chroma: Cached or newly initialized Chroma collection.
        """

        if collection_name not in self._stores:
            self._stores[collection_name] = open_vector_db(
                embeddings=self.embeddings,
                persist_dir=self.persist_dir,
                collection=collection_name,
            )
        return self._stores[collection_name]

    def search(self, *, query: str, collection_name: str, k: int | None = None) -> list[RetrievedChunk]:
        """Search a Chroma collection and convert results into retrieved chunks.

        Args:
            query: Search query used for vector similarity retrieval.
            collection_name: Name of the Chroma collection to search.
            k: Optional number of documents to retrieve.

        Returns:
            list[RetrievedChunk]: Retrieved document chunks with normalized similarity scores.

        Raises:
            VectorStoreError: If vector search fails.
        """

        k = k or self.settings.vector_top_k
        try:
            store = self.get_store(collection_name)
            results = store.similarity_search_with_score(query, k=k)
            return [
                RetrievedChunk(
                    content=doc.page_content,
                    similarity=distance_to_similarity(score),
                    source="vector_db",
                    title=(doc.metadata or {}).get("title"),
                    source_id=(doc.metadata or {}).get("source_id"),
                    collection=collection_name,
                    metadata=doc.metadata or {},
                )
                for doc, score in results
            ]
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(str(exc)) from exc

    def search_with_fallback(self, *, query: str, route: RouteDecision) -> list[RetrievedChunk]:
        """Search the routed collection and fallback to the default collection when empty.

        Args:
            query: Search query used for vector similarity retrieval.
            route: Route decision containing the selected collection.

        Returns:
            list[RetrievedChunk]: Retrieved document chunks from the selected or fallback collection.
        """

        docs = self.search(query=query, collection_name=route.collection)
        if docs:
            return docs
        if route.collection != self.settings.default_collection:
            self.logger.warning(
                "selected collection returned no documents; falling back to default collection",
                extra={
                    "event": "vector_collection_empty",
                    "selected_collection": route.collection,
                    "collection": self.settings.default_collection,
                },
            )
            return self.search(query=query, collection_name=self.settings.default_collection)
        return docs

    def upsert_mcp_results(self, *, route: RouteDecision, results: Iterable[MCPFetchResult]) -> int:
        """Cache MCP fetch results into the routed Chroma collection.

        Args:
            route: Route decision containing the target collection.
            results: MCP fetch results to convert into Chroma documents.

        Returns:
            int: Number of MCP result documents inserted into the vector store.
        """

        docs: list[Document] = []
        ids: list[str] = []
        for result in results:
            doc_id = stable_id(route.collection, result.source_type, result.raw_id, result.content[:120])
            ids.append(doc_id)
            docs.append(
                Document(
                    page_content=result.content,
                    metadata={
                        "title": result.title,
                        "source_id": result.raw_id,
                        "source_type": result.source_type,
                        "collection": route.collection,
                        **result.metadata,
                    },
                )
            )
        if not docs:
            return 0
        store = self.get_store(route.collection)
        try:
            store.delete(ids=ids)
        except Exception:
            pass
        store.add_documents(documents=docs, ids=ids)
        return len(docs)
