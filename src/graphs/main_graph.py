from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from graphs.state import AgentState
from utils.logging_utils import get_logger
from router import route_question
from data_structure.schemas import RouteDecision


def build_main_graph(*, model, settings, checkpointer, memory_subgraph, legal_rag_subgraph, memory_repo):
    """Build and compile the main agent graph.

    Args:
        model: LLM client or runnable used for question routing.
        settings: Runtime settings used by routing and subgraph execution.
        checkpointer: LangGraph checkpointer used to persist graph execution state.
        memory_subgraph: Compiled subgraph used to load relevant user memories.
        legal_rag_subgraph: Compiled subgraph used to retrieve evidence and generate legal answers.
        memory_repo: Repository used to persist user question-answer turns.

    Returns:
        CompiledStateGraph: A compiled LangGraph runnable for the full agent workflow.
    """

    logger = get_logger(__name__)

    async def normalize_and_route(state: AgentState):
        """Normalize the user question and route it to the appropriate legal domain.

        Args:
            state: Current agent graph state containing the raw question and trace metadata.

        Returns:
            dict: State updates containing the normalized question and serialized route decision.
        """

        question = (state.get("question") or "").strip()
        route = await route_question(question, model, settings)
        logger.info(
            "question routed",
            extra={
                "trace_id": state.get("trace_id"),
                "thread_id": state.get("thread_id"),
                "user_id": state.get("user_id"),
                "event": "route_question",
                "route": route.source_type,
                "collection": route.collection,
            },
        )
        return {
            "normalized_question": question,
            "route": route.model_dump(mode="json"),
        }

    async def memory_wrapper(state: AgentState, config: RunnableConfig):
        """Invoke the memory subgraph to load relevant user memories.

        Args:
            state: Current agent graph state containing user, question, and route information.
            config: Runtime configuration passed through to the memory subgraph.

        Returns:
            dict: State updates containing serialized memory items.
        """

        result = await memory_subgraph.ainvoke(
            {
                "user_id": state["user_id"],
                "question": state["normalized_question"],
                "route": state["route"],
            },
            config=config,
        )
        return {"memories": result.get("memories", [])}

    async def legal_rag_wrapper(state: AgentState, config: RunnableConfig):
        """Invoke the legal RAG subgraph and normalize its output into agent state.

        Args:
            state: Current agent graph state containing the normalized question, route, and memories.
            config: Runtime configuration passed through to the legal RAG subgraph.

        Returns:
            dict: State updates containing retrieved evidence, answer text, recovery metadata, and citation validation.
        """

        result = await legal_rag_subgraph.ainvoke(
            {
                "question": state["normalized_question"],
                "route": state["route"],
                "memories": state.get("memories", []),
            },
            config=config,
        )
        rewritten_query = result.get("rewritten_query", "")
        iterations = result.get("iteration", 0)
        fallback_history = result.get("fallback_history", [])

        logger.info(
            "legal RAG subgraph completed",
            extra={
                "trace_id": state.get("trace_id"),
                "event": "rag_completed",
                "rewritten_query": rewritten_query,
                "retrieval_count": len(result.get("retrieved_docs", [])),
                "sufficiency_decision": result.get("retrieval_sufficient", False),
                "fallback_iteration": iterations,
                "mcp_called": result.get("used_mcp", False),
            },
        )
        return {
            "retrieved_docs": result.get("retrieved_docs", []),
            "retrieval_sufficient": result.get("retrieval_sufficient", False),
            "used_mcp": result.get("used_mcp", False),
            "answer": result.get("answer", ""),
            "rewritten_query": rewritten_query,
            "retrieval_iterations": iterations,
            "fallback_history": fallback_history,
            "sufficiency_reason": result.get("sufficiency_reason", ""),
            "recovery_steps": result.get("recovery_steps", []),
            "evidence_list": result.get("evidence_list", []),
            "citation_validation": result.get("citation_validation", {}),
        }

    async def persist_turn_memory(state: AgentState):
        """Persist the completed question-answer turn into memory storage.

        Args:
            state: Current agent graph state containing the normalized question, route, answer, and user ID.

        Returns:
            dict: Empty state update after recording the turn.
        """

        route = RouteDecision.model_validate(state["route"])
        memory_repo.record_turn(
            user_id=state["user_id"],
            question=state["normalized_question"],
            route=route,
            answer=state.get("answer", ""),
        )
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("normalize_and_route", normalize_and_route)
    builder.add_node("memory_wrapper", memory_wrapper)
    builder.add_node("legal_rag_wrapper", legal_rag_wrapper)
    builder.add_node("persist_turn_memory", persist_turn_memory)
    builder.add_edge(START, "normalize_and_route")
    builder.add_edge("normalize_and_route", "memory_wrapper")
    builder.add_edge("memory_wrapper", "legal_rag_wrapper")
    builder.add_edge("legal_rag_wrapper", "persist_turn_memory")
    builder.add_edge("persist_turn_memory", END)
    return builder.compile(checkpointer=checkpointer)
