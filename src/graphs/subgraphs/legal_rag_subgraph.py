from langgraph.graph import END, START, StateGraph

from answering import generate_answer
from utils.config import Settings
from utils.exceptions import ConfigError, MCPError
from llm_model.llm import ainvoke_json
from data_structure.schemas import MemoryItem, RetrievedChunk, RouteDecision

from graphs.state import LegalRAGState


async def judge_sufficiency(*, model, settings: Settings, question: str, docs: list[RetrievedChunk]) -> bool:
    if len(docs) < settings.min_retrieved_docs:
        return False
    if docs and max(doc.similarity for doc in docs) < settings.similarity_threshold:
        return False

    preview = "\n\n".join(f"[{idx}] {doc.content[:400]}" for idx, doc in enumerate(docs[:3], start=1))
    payload = await ainvoke_json(
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a RAG sufficiency judge. Decide whether the provided documents alone "
                    "are enough to answer the core question. "
                    'Reply only in JSON: {"sufficient": true|false, "reason": "..."}'
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nDocument preview:\n{preview}"},
        ],
        default={"sufficient": False, "reason": "fallback"},
    )
    return bool(payload.get("sufficient", False))


def build_legal_rag_subgraph(*, model, settings: Settings, vector_store, mcp_gateway=None):
    async def retrieve_cache(state: LegalRAGState):
        route = RouteDecision.model_validate(state["route"])
        docs = vector_store.search_with_fallback(query=state["question"], route=route)
        return {"retrieved_docs": [doc.model_dump(mode="json") for doc in docs]}

    async def judge_cache(state: LegalRAGState):
        docs = [RetrievedChunk.model_validate(item) for item in state.get("retrieved_docs", [])]
        sufficient = await judge_sufficiency(
            model=model,
            settings=settings,
            question=state["question"],
            docs=docs,
        )
        return {"retrieval_sufficient": sufficient}

    async def fetch_mcp_and_cache(state: LegalRAGState):
        route = RouteDecision.model_validate(state["route"])
        if mcp_gateway is None or not mcp_gateway.is_enabled():
            return {
                "used_mcp": False,
                "retrieval_sufficient": False,
                "retrieved_docs": state.get("retrieved_docs", []),
            }
        try:
            results = await mcp_gateway.search_and_fetch(
                route=route,
                query=state["question"],
                fetch_top_n=settings.mcp_fetch_top_n,
            )
            vector_store.upsert_mcp_results(route=route, results=results)
            docs = vector_store.search_with_fallback(query=state["question"], route=route)
            return {
                "retrieved_docs": [doc.model_dump(mode="json") for doc in docs],
                "used_mcp": True,
                "retrieval_sufficient": bool(docs),
            }
        except (ConfigError, MCPError):
            return {
                "used_mcp": False,
                "retrieval_sufficient": False,
                "retrieved_docs": state.get("retrieved_docs", []),
            }

    async def answer(state: LegalRAGState):
        route = RouteDecision.model_validate(state["route"])
        memories = [MemoryItem.model_validate(item) for item in state.get("memories", [])]
        docs = [RetrievedChunk.model_validate(item) for item in state.get("retrieved_docs", [])]
        answer_text = await generate_answer(
            model=model,
            settings=settings,
            question=state["question"],
            route=route,
            memories=memories,
            docs=docs,
        )
        return {"answer": answer_text}

    def route_after_judge(state: LegalRAGState):
        return "answer" if state.get("retrieval_sufficient") else "fetch_mcp_and_cache"

    builder = StateGraph(LegalRAGState)
    builder.add_node("retrieve_cache", retrieve_cache)
    builder.add_node("judge_cache", judge_cache)
    builder.add_node("fetch_mcp_and_cache", fetch_mcp_and_cache)
    builder.add_node("answer", answer)
    builder.add_edge(START, "retrieve_cache")
    builder.add_edge("retrieve_cache", "judge_cache")
    builder.add_conditional_edges(
        "judge_cache",
        route_after_judge,
        {
            "answer": "answer",
            "fetch_mcp_and_cache": "fetch_mcp_and_cache",
        },
    )
    builder.add_edge("fetch_mcp_and_cache", "answer")
    builder.add_edge("answer", END)
    return builder.compile()
