from langgraph.graph import END, START, StateGraph

from answering import generate_answer
from utils.config import Settings
from utils.exceptions import ConfigError, MCPError
from utils.logging_utils import get_logger
from llm_model.llm import ainvoke_json
from data_structure.schemas import MemoryItem, RecoveryStep, RetrievedChunk, RouteDecision, SufficiencyDecision
from router import rewrite_query_for_retrieval, refine_query_for_retry
from utils.citations import attach_evidence_ids, validate_citations

from graphs.state import LegalRAGState


def docs_observation(docs: list[RetrievedChunk]) -> dict:
    """Summarize a retrieved document set for recovery traces and logging.

    Args:
        docs: Retrieved chunks to summarize.

    Returns:
        dict: A dictionary containing document count, top similarity score, distinct source names, and distinct collection names.
    """

    return {
        "doc_count": len(docs),
        "top_score": max((doc.similarity for doc in docs), default=0.0),
        "sources": sorted({doc.source for doc in docs if doc.source}),
        "collections": sorted({doc.collection for doc in docs if doc.collection}),
    }


async def judge_sufficiency(
    *, model, settings: Settings, question: str, docs: list[RetrievedChunk]
) -> SufficiencyDecision:
    """Judge whether retrieved legal evidence is sufficient to answer the question.

    Args:
        model: LLM client or runnable used for JSON invocation.
        settings: Runtime settings containing retrieval thresholds and sufficiency criteria.
        question: Original user question.
        docs: Retrieved legal evidence chunks.

    Returns:
        SufficiencyDecision: A decision containing sufficiency status, reason, and suggested recovery action.
    """

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
    """Rerank retrieved documents by LLM-estimated relevance to the question.

    Args:
        model: LLM client or runnable used for JSON invocation.
        question: Original user question.
        docs: Retrieved chunks in vector-search order.
        top_k: Number of documents to prioritize after reranking.

    Returns:
        list[RetrievedChunk]: Retrieved chunks reordered by relevance score.
    """

    if len(docs) <= 1:
        return docs

    candidates = docs[: top_k * 2]

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
                    'Reply only in JSON: {"scores": [7, 3, 9, ...]}\n'
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
    remaining = [doc for doc in docs[len(candidates) :]]
    reranked = [doc for _, doc in scored[:top_k]] + remaining
    return reranked


def build_legal_rag_subgraph(*, model, settings: Settings, vector_store, mcp_gateway=None):
    """Build and compile the legal RAG retrieval subgraph.

    Args:
        model: LLM client or runnable used across retrieval, reranking, judging, fallback, and answer generation.
        settings: Runtime settings for retrieval thresholds, reranking, MCP usage, and retry limits.
        vector_store: Vector search backend used for legal evidence retrieval.
        mcp_gateway: Optional MCP gateway used for external legal evidence augmentation.

    Returns:
        CompiledStateGraph: A compiled LangGraph runnable for the legal RAG workflow.
    """

    logger = get_logger(__name__)

    async def rewrite_query(state: LegalRAGState):
        """Rewrite the original question into a retrieval-oriented query.

        Args:
            state: Current legal RAG graph state.

        Returns:
            dict: State updates containing the rewritten query and initialized recovery metadata.
        """

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
            "recovery_steps": [],
            "used_mcp": False,
        }

    async def retrieve(state: LegalRAGState):
        """Retrieve legal evidence from the vector store and merge it into state.

        Args:
            state: Current legal RAG graph state.

        Returns:
            dict: State updates containing merged retrieved documents and recovery step records.
        """

        route = RouteDecision.model_validate(state["route"])
        query = state.get("rewritten_query") or state["question"]
        docs = vector_store.search_with_fallback(query=query, route=route)

        existing = [RetrievedChunk.model_validate(d) for d in state.get("retrieved_docs", [])]
        before_count = len(existing)
        seen_keys = {(d.source_id, d.content[:120]) for d in existing}

        for doc in docs:
            key = (doc.source_id, doc.content[:120])
            if key not in seen_keys:
                existing.append(doc)
                seen_keys.add(key)

        top_score = max((d.similarity for d in existing), default=0.0)
        recovery_steps = list(state.get("recovery_steps", []))
        iteration = state.get("iteration", 0)
        if iteration > 0:
            recovery_steps.append(
                RecoveryStep(
                    iteration=iteration,
                    reasoning=state.get("sufficiency_reason", "previous retrieval was insufficient"),
                    action="vector_research",
                    action_input={"query": query, "collection": route.collection},
                    observation=docs_observation(docs),
                    evidence_delta=max(len(existing) - before_count, 0),
                    next_query=query,
                    selected_collection=route.collection,
                    source="chroma",
                ).model_dump(mode="json")
            )
            if len(existing) > before_count:
                recovery_steps.append(
                    RecoveryStep(
                        iteration=iteration,
                        reasoning="new vector evidence was added to the working context",
                        action="merge_evidence",
                        action_input={"previous_doc_count": before_count, "new_doc_count": len(docs)},
                        observation=docs_observation(existing),
                        evidence_delta=len(existing) - before_count,
                        next_query=query,
                        selected_collection=route.collection,
                        source="state",
                    ).model_dump(mode="json")
                )
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
        return {
            "retrieved_docs": [d.model_dump(mode="json") for d in existing],
            "recovery_steps": recovery_steps,
        }

    async def rerank(state: LegalRAGState):
        """Rerank retrieved evidence before sufficiency judgment.

        Args:
            state: Current legal RAG graph state.

        Returns:
            dict: State updates containing reranked retrieved documents.
        """

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
        """Judge whether the current retrieved evidence can support answer generation.

        Args:
            state: Current legal RAG graph state.

        Returns:
            dict: State updates containing sufficiency status, reason, and suggested fallback action.
        """

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
        return {
            "retrieval_sufficient": decision.sufficient,
            "sufficiency_reason": decision.reason,
            "suggested_action": decision.suggested_action,
        }

    def route_after_judge(state: LegalRAGState) -> str:
        """Select the next graph node after sufficiency judgment.

        Args:
            state: Current legal RAG graph state.

        Returns:
            str: The next route name, either "answer" or "fallback".
        """

        if state.get("retrieval_sufficient"):
            return "answer"
        iteration = state.get("iteration", 0)
        if iteration >= settings.max_retrieval_iterations:
            return "answer"
        return "fallback"

    async def fallback(state: LegalRAGState):
        """Choose and execute a recovery action when retrieved evidence is insufficient.

        Args:
            state: Current legal RAG graph state.

        Returns:
            dict: State updates for the next retrieval attempt, including fallback history and recovery steps.
        """

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
        reasoning = state.get("sufficiency_reason", "retrieved evidence was judged insufficient")
        result: dict = {
            "iteration": iteration + 1,
            "fallback_history": history,
        }
        recovery_steps = list(state.get("recovery_steps", []))

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
            doc_summaries = "\n".join(f"- {d.title or 'N/A'}: {d.content[:150]}" for d in existing_docs[:3])
            refined = await refine_query_for_retry(question, current_query, route, doc_summaries, model)
            result["rewritten_query"] = refined
            recovery_steps.append(
                RecoveryStep(
                    iteration=iteration + 1,
                    reasoning=reasoning,
                    action="refine_query",
                    action_input={"question": question, "previous_query": current_query},
                    observation={"refined_query": refined, "current_docs": len(existing_docs)},
                    evidence_delta=0,
                    next_query=refined,
                    selected_collection=route.collection,
                    source="llm",
                ).model_dump(mode="json")
            )
            logger.info(
                "query refined for retry",
                extra={"event": "query_refine", "rewritten_query": refined},
            )

        elif action == "broaden_collection":
            new_route = route.model_copy(update={"collection": settings.default_collection})
            result["route"] = new_route.model_dump(mode="json")
            recovery_steps.append(
                RecoveryStep(
                    iteration=iteration + 1,
                    reasoning=reasoning,
                    action="broaden_collection",
                    action_input={"from_collection": route.collection, "to_collection": settings.default_collection},
                    observation={"selected_collection": settings.default_collection},
                    evidence_delta=0,
                    next_query=current_query,
                    selected_collection=settings.default_collection,
                    source="router",
                ).model_dump(mode="json")
            )
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
                recovery_steps.append(
                    RecoveryStep(
                        iteration=iteration + 1,
                        reasoning=reasoning,
                        action="mcp_augmentation",
                        action_input={"query": question, "route": route.model_dump(mode="json")},
                        observation=docs_observation(mcp_docs),
                        evidence_delta=len(mcp_docs),
                        next_query=current_query,
                        selected_collection=route.collection,
                        source="mcp",
                    ).model_dump(mode="json")
                )
                recovery_steps.append(
                    RecoveryStep(
                        iteration=iteration + 1,
                        reasoning="MCP evidence was merged into retrieved context",
                        action="merge_evidence",
                        action_input={"previous_doc_count": len(existing_docs), "mcp_doc_count": len(mcp_docs)},
                        observation=docs_observation(merged),
                        evidence_delta=len(mcp_docs),
                        next_query=current_query,
                        selected_collection=route.collection,
                        source="state",
                    ).model_dump(mode="json")
                )

                logger.info(
                    "MCP augmentation completed",
                    extra={
                        "event": "mcp_augmentation",
                        "mcp_called": True,
                        "retrieval_count": len(mcp_docs),
                    },
                )

                try:
                    upserted = vector_store.upsert_mcp_results(route=route, results=mcp_results)
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
                recovery_steps.append(
                    RecoveryStep(
                        iteration=iteration + 1,
                        reasoning=reasoning,
                        action="mcp_augmentation",
                        action_input={"query": question, "route": route.model_dump(mode="json")},
                        observation="MCP augmentation failed or is not configured",
                        evidence_delta=0,
                        next_query=current_query,
                        selected_collection=route.collection,
                        source="mcp",
                    ).model_dump(mode="json")
                )
                logger.warning(
                    "MCP augmentation failed, continuing without",
                    extra={"event": "mcp_augmentation_failed", "mcp_called": True},
                    exc_info=True,
                )

        result["recovery_steps"] = recovery_steps
        return result

    async def answer(state: LegalRAGState):
        """Generate the final legal answer from retrieved evidence and memory context.

        Args:
            state: Current legal RAG graph state.

        Returns:
            dict: State updates containing the answer, evidence list, and citation validation result.
        """

        route = RouteDecision.model_validate(state["route"])
        memories = [MemoryItem.model_validate(item) for item in state.get("memories", [])]
        docs = [RetrievedChunk.model_validate(item) for item in state.get("retrieved_docs", [])]
        is_sufficient = state.get("retrieval_sufficient", False)
        docs, evidence_list = attach_evidence_ids(docs)

        answer_text = await generate_answer(
            model=model,
            settings=settings,
            question=state["question"],
            route=route,
            memories=memories,
            docs=docs,
            evidence_sufficient=is_sufficient,
        )
        citation_validation = validate_citations(answer_text, evidence_list)
        logger.info(
            "answer generated",
            extra={
                "event": "answer_generation",
                "retrieval_count": len(docs),
                "recovery_steps": len(state.get("recovery_steps", [])),
                "citation_valid": citation_validation.get("valid"),
            },
        )
        return {
            "answer": answer_text,
            "retrieved_docs": [doc.model_dump(mode="json") for doc in docs],
            "evidence_list": evidence_list,
            "citation_validation": citation_validation,
        }

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
