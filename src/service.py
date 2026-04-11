import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from utils.config import Settings, get_settings
from llm_model.llm import LLMFactory
from utils.logging_utils import configure_logging, get_logger, new_trace_id
from mcp_client import KoreanLawMCPGateway
from storages.memory_store import MemoryRepository
from graphs.main_graph import build_main_graph
from graphs.subgraphs.legal_rag_subgraph import build_legal_rag_subgraph
from graphs.subgraphs.memory_subgraph import build_memory_subgraph
from data_structure.schemas import AnswerResult, MemoryItem, RetrievedChunk, RouteDecision
from storages.vector_store import LegalVectorStore


class LegalAgentService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)
        self.logger = get_logger(__name__)
        self._graph_lock = asyncio.Lock()
        self._health_lock = asyncio.Lock()
        self._health_checked = False

        factory = LLMFactory(self.settings)
        factory.validate_settings()
        self.llm_factory = factory
        self.chat_model = factory.create_chat_model()
        self.embeddings = factory.create_embeddings()
        self._log_startup_configuration()

        self.memory_repo = MemoryRepository(
            db_path=self.settings.memory_path,
            half_life_days=self.settings.memory_half_life_days,
            default_importance=self.settings.memory_default_importance,
        )
        self.vector_store = LegalVectorStore(
            persist_dir=self.settings.chroma_path,
            embeddings=self.embeddings,
            settings=self.settings,
        )
        self.mcp_gateway = KoreanLawMCPGateway(self.settings)
        self.checkpointer_cm = None
        self.checkpointer = None
        self.memory_subgraph = build_memory_subgraph(self.memory_repo, self.settings)
        self.legal_rag_subgraph = build_legal_rag_subgraph(
            model=self.chat_model,
            settings=self.settings,
            vector_store=self.vector_store,
            mcp_gateway=self.mcp_gateway,
        )
        self.graph = None

    def _log_startup_configuration(self) -> None:
        self.logger.info(
            "LLM backend configuration",
            extra={
                "event": "startup_config",
                "provider": self.settings.local_llm_provider,
                "model": self.settings.local_llm_model,
                "base_url": self.settings.local_llm_base_url,
                "embedding_provider": self.settings.local_embedding_provider,
                "embedding_model": self.settings.local_embedding_model,
            },
        )

    async def _ensure_healthcheck(self) -> None:
        if self._health_checked:
            return
        async with self._health_lock:
            if self._health_checked:
                return
            await self.llm_factory.acheck_chat_backend()
            self._health_checked = True
            self.logger.info(
                "LLM backend health check passed",
                extra={
                    "event": "llm_healthcheck",
                    "provider": self.settings.local_llm_provider,
                    "model": self.settings.local_llm_model,
                },
            )

    async def _ensure_graph(self) -> None:
        if self.graph is not None:
            return
        async with self._graph_lock:
            if self.graph is not None:
                return
            self.checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(self.settings.checkpoint_path))
            self.checkpointer = await self.checkpointer_cm.__aenter__()
            await self.checkpointer.setup()
            self.graph = build_main_graph(
                model=self.chat_model,
                settings=self.settings,
                checkpointer=self.checkpointer,
                memory_subgraph=self.memory_subgraph,
                legal_rag_subgraph=self.legal_rag_subgraph,
                memory_repo=self.memory_repo,
            )

    async def aask(self, *, question: str, user_id: str, thread_id: str) -> AnswerResult:
        await self._ensure_healthcheck()
        await self._ensure_graph()
        trace_id = new_trace_id()
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
                "user_id": user_id,
                "thread_id": thread_id,
                "trace_id": trace_id,
                "question": question,
            },
            config=config,
        )
        return AnswerResult(
            answer=state.get("answer", ""),
            route=RouteDecision.model_validate(state.get("route", {})),
            memories=[MemoryItem.model_validate(item) for item in state.get("memories", [])],
            retrieved_docs=[RetrievedChunk.model_validate(item) for item in state.get("retrieved_docs", [])],
            used_mcp=bool(state.get("used_mcp", False)),
            trace_id=trace_id,
        )

    async def aclose(self) -> None:
        if self.checkpointer_cm is None:
            return
        try:
            await self.checkpointer_cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.checkpointer_cm = None
            self.checkpointer = None
            self.graph = None
