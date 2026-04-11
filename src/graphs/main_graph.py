from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from graphs.state import AgentState
from utils.logging_utils import get_logger
from router import route_question
from data_structure.schemas import RouteDecision


def build_main_graph(*, model, settings, checkpointer, memory_subgraph, legal_rag_subgraph, memory_repo):
    logger = get_logger(__name__)

    async def normalize_and_route(state: AgentState):
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
        result = await legal_rag_subgraph.ainvoke(
            {
                "question": state["normalized_question"],
                "route": state["route"],
                "memories": state.get("memories", []),
            },
            config=config,
        )
        return {
            "retrieved_docs": result.get("retrieved_docs", []),
            "retrieval_sufficient": result.get("retrieval_sufficient", False),
            "used_mcp": result.get("used_mcp", False),
            "answer": result.get("answer", ""),
        }

    async def persist_turn_memory(state: AgentState):
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
