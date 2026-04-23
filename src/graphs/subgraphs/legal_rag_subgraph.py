from langgraph.graph import END, START, StateGraph

from answering import generate_answer
from utils.config import Settings
from utils.exceptions import ConfigError, MCPError
from utils.logging_utils import get_logger
from llm_model.llm import ainvoke_json
from data_structure.schemas import MemoryItem, RetrievedChunk, RouteDecision, SufficiencyDecision
from router import rewrite_query_for_retrieval, refine_query_for_retry

from graphs.state import LegalRAGState


async def judge_sufficiency(*, model, settings: Settings, question: str, docs: list[RetrievedChunk]) -> SufficiencyDecision:
    if len(docs) < settings.min_retrieved_docs:
        return SufficiencyDecision(
            sufficient=False,
            reason=f"too few docs ({len(docs)} < {settings.min_retrieved_docs})",
            suggested_action="broaden_search",
        )
    if docs and max(doc.similarity for doc in docs) < settings.similarity_threshold:
        return SufficiencyDecision(
            sufficient=False,
            reason=f"top score {max(doc.similarity for doc in docs):.3f} below threshold {settings.similarity_threshold}",
            suggested_action="refine_query",
        )

    preview = "\n\n".join(f"[{idx}] {doc.content[:400]}" for idx, doc in enumerate(docs[:3], start=1))
    payload = await ainvoke_json(
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a RAG sufficiency judge for Korean legal QA. "
                    "Decide whether the provided documents contain enough evidence "
                    "to answer the core legal question.\n"
                    "Consider:\n"
                    "- Do the documents address the specific legal issue asked?\n"
                    "- Are relevant statutes, articles, or precedents present?\n"
                    "- Is there enough detail for a substantive answer?\n"
                    'Reply only in JSON: {"sufficient": true|false, "reason": "...", "suggested_action": "none|refine_query|broaden_collection|mcp_augmentation"}'
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nDocument preview:\n{preview}"},
        ],
        default={"sufficient": False, "reason": "fallback", "suggested_action": "refine_query"},
    )
    return SufficiencyDecision(
        sufficient=bool(payload.get("sufficient", False)),
        reason=str(payload.get("reason", "unknown")),
        suggested_action=str(payload.get("suggested_action", "refine_query")),
    )


async def rerank_documents(*, model, question: str, docs: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if len(docs) <= 1:
        return docs

    candidates = docs[:top_k * 2]

    doc_list = "\n\n".join(
        f"[Doc {idx}] title={doc.title or 'N/A'} source={doc.source}\n{doc.content[:300]}"
        for idx, doc in enumerate(candidates)
    )
    payload = await ainvoke_json(
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Korean legal document relevance scorer. "
                    "Score each document's relevance to the question on a 0-10 scale.\n"
                    "Consider:\n"
                    "- Does the document directly address the legal issue?\n"
                    "- Does it contain applicable statutes or precedents?\n"
                    "- Is the content specific enough to be useful?\n"
                    "Reply only in JSON: {\"scores\": [7, 3, 9, ...]}\n"
                    "The scores array must have exactly one score per document, in order."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocuments:\n{doc_list}",
            },
        ],
        default={"scores": []},
    )

    raw_scores = payload.get("scores", [])
    if not isinstance(raw_scores, list) or len(raw_scores) != len(candidates):
        return docs[:top_k]

    scored = []
    for doc, score in zip(candidates, raw_scores):
        try:
            relevance = float(score)
        except (TypeError, ValueError):
            relevance = 0.0
        scored.append((relevance, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    remaining = [doc for doc in docs[len(candidates):]]
    reranked = [doc for _, doc in scored[:top_k]] + remaining
    return reranked


def build_legal_rag_subgraph(*, model, settings: Settings, vector_store, mcp_gateway=None):
    logger = get_logger(__name__)

    async def rewrite_query(state: LegalRAGState):
        route = RouteDecision.model_validate(state["route"])
        question = state["question"]
        rewritten = await rewrite_query_for_retrieval(question, route, model)

        logger.info(
            "query rewritten for retrieval",
            extra={
                "event": "query_rewrite",
                "rewritten_query": rewritten,
                "source_type": route.source_type,
                "topic": route.topic,
            },
        )
        return {
            "rewritten_query": rewritten,
            "iteration": 0,
            "fallback_history": [],
        }

    async def retrieve(state: LegalRAGState):
        route = RouteDecision.model_validate(state["route"])
        query = state.get("rewritten_query") or state["question"]
        docs = vector_store.search_with_fallback(query=query, route=route)

        existing = [RetrievedChunk.model_validate(d) for d in state.get("retrieved_docs", [])]
        seen_keys = {(d.source_id, d.content[:120]) for d in existing}

        for doc in docs:
            key = (doc.source_id, doc.content[:120])
            if key not in seen_keys:
                existing.append(doc)
                seen_keys.add(key)

        top_score = max((d.similarity for d in existing), default=0.0)
        logger.info(
            "vector retrieval completed",
            extra={
                "event": "retrieve",
                "selected_collection": route.collection,
                "retrieval_count": len(existing),
                "top_score": round(top_score, 4),
                "fallback_iteration": state.get("iteration", 0),
            },
        )
        return {"retrieved_docs": [d.model_dump(mode="json") for d in existing]}

    async def rerank(state: LegalRAGState):
        docs = [RetrievedChunk.model_validate(d) for d in state.get("retrieved_docs", [])]
        question = state["question"]
        reranked = await rerank_documents(
            model=model,
            question=question,
            docs=docs,
            top_k=settings.rerank_top_k,
        )

        logger.info(
            "reranking completed",
            extra={
                "event": "rerank",
                "rerank_scores": [round(d.similarity, 4) for d in reranked[:5]],
                "retrieval_count": len(reranked),
            },
        )
        return {"retrieved_docs": [d.model_dump(mode="json") for d in reranked]}

    async def judge(state: LegalRAGState):
        docs = [RetrievedChunk.model_validate(d) for d in state.get("retrieved_docs", [])]
        decision = await judge_sufficiency(
            model=model,
            settings=settings,
            question=state["question"],
            docs=docs,
        )

        logger.info(
            "sufficiency judged",
            extra={
                "event": "sufficiency_judge",
                "sufficiency_decision": decision.sufficient,
                "sufficiency_reason": decision.reason,
                "fallback_iteration": state.get("iteration", 0),
            },
        )
        return {"retrieval_sufficient": decision.sufficient}

    def route_after_judge(state: LegalRAGState) -> str:
        if state.get("retrieval_sufficient"):
            return "answer"
        iteration = state.get("iteration", 0)
        if iteration >= settings.max_retrieval_iterations:
            return "answer"
        return "fallback"

    async def fallback(state: LegalRAGState):
        iteration = state.get("iteration", 0)
        history = list(state.get("fallback_history", []))
        route = RouteDecision.model_validate(state["route"])
        question = state["question"]
        current_query = state.get("rewritten_query") or question
        existing_docs = [RetrievedChunk.model_validate(d) for d in state.get("retrieved_docs", [])]

        if "refine_query" not in history:
            action = "refine_query"
        elif "broaden_collection" not in history and route.collection != settings.default_collection:
            action = "broaden_collection"
        elif "mcp_augmentation" not in history and mcp_gateway is not None and mcp_gateway.is_enabled():
            action = "mcp_augmentation"
        else:
            action = "refine_query"

        history.append(action)
        result: dict = {
            "iteration": iteration + 1,
            "fallback_history": history,
        }

        logger.info(
            "fallback loop entered",
            extra={
                "event": "fallback",
                "fallback_action": action,
                "fallback_iteration": iteration + 1,
                "selected_collection": route.collection,
            },
        )

        if action == "refine_query":
            doc_summaries = "\n".join(
                f"- {d.title or 'N/A'}: {d.content[:150]}" for d in existing_docs[:3]
            )
            refined = await refine_query_for_retry(
                question, current_query, route, doc_summaries, model
            )
            result["rewritten_query"] = refined
            logger.info(
                "query refined for retry",
                extra={"event": "query_refine", "rewritten_query": refined},
            )

        elif action == "broaden_collection":
            new_route = route.model_copy(update={"collection": settings.default_collection})
            result["route"] = new_route.model_dump(mode="json")
            logger.info(
                "collection broadened",
                extra={
                    "event": "broaden_collection",
                    "selected_collection": settings.default_collection,
                },
            )

        elif action == "mcp_augmentation":
            result["used_mcp"] = False
            try:
                mcp_results = await mcp_gateway.search_and_fetch(
                    route=route,
                    query=question,
                    fetch_top_n=settings.mcp_fetch_top_n,
                )
                mcp_docs = [
                    RetrievedChunk(
                        content=r.content,
                        similarity=0.0,
                        source="mcp_augmentation",
                        title=r.title,
                        source_id=r.raw_id,
                        collection=route.collection,
                        metadata={**r.metadata, "fetched_via": "mcp_augmentation"},
                    )
                    for r in mcp_results
                ]
                merged = list(existing_docs) + mcp_docs
                result["retrieved_docs"] = [d.model_dump(mode="json") for d in merged]
                result["used_mcp"] = True

                logger.info(
                    "MCP augmentation completed",
                    extra={
                        "event": "mcp_augmentation",
                        "mcp_called": True,
                        "retrieval_count": len(mcp_docs),
                    },
                )

                try:
                    upserted = vector_store.upsert_mcp_results(
                        route=route, results=mcp_results
                    )
                    logger.info(
                        "MCP results cached to vector store",
                        extra={"event": "mcp_upsert", "mcp_upserted": upserted},
                    )
                except Exception:
                    logger.warning(
                        "failed to upsert MCP results to vector store",
                        extra={"event": "mcp_upsert_failed"},
                        exc_info=True,
                    )
            except (ConfigError, MCPError):
                logger.warning(
                    "MCP augmentation failed, continuing without",
                    extra={"event": "mcp_augmentation_failed", "mcp_called": True},
                    exc_info=True,
                )

        return result

    async def answer(state: LegalRAGState):
        route = RouteDecision.model_validate(state["route"])
        memories = [MemoryItem.model_validate(item) for item in state.get("memories", [])]
        docs = [RetrievedChunk.model_validate(item) for item in state.get("retrieved_docs", [])]
        is_sufficient = state.get("retrieval_sufficient", False)

        answer_text = await generate_answer(
            model=model,
            settings=settings,
            question=state["question"],
            route=route,
            memories=memories,
            docs=docs,
            evidence_sufficient=is_sufficient,
        )
        return {"answer": answer_text}

    builder = StateGraph(LegalRAGState)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("judge", judge)
    builder.add_node("fallback", fallback)
    builder.add_node("answer", answer)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "judge")
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )
    builder.add_edge("fallback", "retrieve")
    builder.add_edge("answer", END)

    return builder.compile()
