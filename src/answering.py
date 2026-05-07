from utils.config import Settings
from llm_model.llm import ainvoke_text
from data_structure.schemas import MemoryItem, RetrievedChunk, RouteDecision
from utils.citations import evidence_label


def truncate(text: str, limit: int) -> str:
    """Truncate text to a maximum character limit with a truncation marker.

    Args:
        text: Source text to truncate.
        limit: Maximum number of characters to keep before adding the truncation marker.

    Returns:
        str: Original text when within the limit, otherwise truncated text with a marker.
    """

    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def group_docs_by_source(docs: list[RetrievedChunk], route: RouteDecision) -> str:
    """Group retrieved legal documents into formatted evidence sections by source type.

    Args:
        docs: Retrieved evidence chunks to group and format.
        route: Route decision used to infer the expected legal source type.

    Returns:
        str: Formatted evidence block grouped into statute, precedent, constitutional, and other sections.
    """

    statute_docs = []
    precedent_docs = []
    constitutional_docs = []
    other_docs = []

    for doc in docs:
        meta_source = doc.metadata.get("source_type", "")
        meta_law = doc.metadata.get("law_name", "")

        if meta_source == "precedent" or route.source_type == "precedent":
            precedent_docs.append(doc)
        elif meta_source == "constitutional" or route.source_type == "constitutional" or "헌법" in meta_law:
            constitutional_docs.append(doc)
        elif meta_source in ("law", "") and route.source_type == "law":
            statute_docs.append(doc)
        else:
            other_docs.append(doc)

    if not statute_docs and not precedent_docs and not constitutional_docs:
        statute_docs = other_docs
        other_docs = []

    sections = []

    if statute_docs:
        block = "\n\n".join(
            f"{evidence_label(d, idx)} [법령 {idx}] title={d.title or 'N/A'} source_id={d.source_id or 'N/A'} "
            f"similarity={d.similarity:.3f}\n{d.content}"
            for idx, d in enumerate(statute_docs, start=1)
        )
        sections.append(f"=== 법령 근거 ===\n{block}")

    if precedent_docs:
        block = "\n\n".join(
            f"{evidence_label(d, idx)} [판례 {idx}] title={d.title or 'N/A'} source_id={d.source_id or 'N/A'} "
            f"similarity={d.similarity:.3f}\n{d.content}"
            for idx, d in enumerate(precedent_docs, start=1)
        )
        sections.append(f"=== 판례 근거 ===\n{block}")

    if constitutional_docs:
        block = "\n\n".join(
            f"{evidence_label(d, idx)} [헌재 {idx}] title={d.title or 'N/A'} source_id={d.source_id or 'N/A'} "
            f"similarity={d.similarity:.3f}\n{d.content}"
            for idx, d in enumerate(constitutional_docs, start=1)
        )
        sections.append(f"=== 헌법/헌재 근거 ===\n{block}")

    if other_docs:
        block = "\n\n".join(
            f"{evidence_label(d, idx)} [기타 {idx}] title={d.title or 'N/A'} source_id={d.source_id or 'N/A'} "
            f"similarity={d.similarity:.3f}\n{d.content}"
            for idx, d in enumerate(other_docs, start=1)
        )
        sections.append(f"=== 기타 근거 ===\n{block}")

    return "\n\n".join(sections)


async def generate_answer(
    *,
    model,
    settings: Settings,
    question: str,
    route: RouteDecision,
    memories: list[MemoryItem],
    docs: list[RetrievedChunk],
    evidence_sufficient: bool = True,
) -> str:
    """Generate a Korean legal answer grounded in retrieved evidence and user memories.

    Args:
        model: LLM client or runnable used for answer generation.
        settings: Runtime settings containing context length limits.
        question: Original user question.
        route: Route decision describing the legal source type, topic, and collection.
        memories: Relevant user memories to include as auxiliary context.
        docs: Retrieved legal evidence chunks to ground the answer.
        evidence_sufficient: Whether the retrieved evidence was judged sufficient for answering.

    Returns:
        str: Generated Korean legal answer with evidence citations and legal-information disclaimer.
    """

    memory_block = (
        "\n\n".join(
            f"- ({idx}) [{memory.domain}/{memory.topic}] {memory.content}"
            for idx, memory in enumerate(memories, start=1)
        )
        or "- None"
    )

    doc_block = group_docs_by_source(docs, route)
    doc_block = truncate(doc_block, settings.max_context_chars)

    sufficiency_note = ""
    if not evidence_sufficient:
        sufficiency_note = (
            "\n\n[IMPORTANT: The retrieved evidence may be insufficient. "
            "Explicitly state uncertainty and limitations in your answer. "
            "Do NOT fabricate information beyond what is provided.]"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant for Korean legal question answering. "
                "Always answer in Korean and base your answer strictly on the provided documents and memories. "
                "Clearly distinguish between 법령(statutes), 판례(precedents), and 헌재결정(constitutional decisions) when citing. "
                "If the evidence is weak or incomplete, explicitly state uncertainty and what information is missing. "
                "Never fabricate legal provisions, case numbers, or precedent details. "
                "Add one final line saying this is general legal information, not professional legal advice."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[Question]\n{question}\n\n"
                f"[Routing]\nsource_type={route.source_type}, topic={route.topic}, collection={route.collection}\n\n"
                f"[Relevant memories]\n{memory_block}\n\n"
                f"[Retrieved documents (grouped by source type)]\n{doc_block}"
                f"{sufficiency_note}\n\n"
                "Requirements:\n"
                "1. Core answer grounded in the provided evidence\n"
                "2. Cite specific statutes, precedents, or constitutional decisions from the documents using evidence IDs like [E1] when applicable\n"
                "3. Distinguish between 법령/판례/헌재결정 perspectives where applicable\n"
                "4. State limitations if evidence is insufficient\n"
                "5. Put this exact sentence on the last line: "
                "'※ 본 답변은 일반적인 법률 정보 제공이며 구체적 사건은 전문가 검토가 필요할 수 있습니다.'"
            ),
        },
    ]
    return await ainvoke_text(model, messages)
