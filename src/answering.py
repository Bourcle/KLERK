from utils.config import Settings
from llm_model.llm import ainvoke_text
from data_structure.schemas import MemoryItem, RetrievedChunk, RouteDecision


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


async def generate_answer(
    *,
    model,
    settings: Settings,
    question: str,
    route: RouteDecision,
    memories: list[MemoryItem],
    docs: list[RetrievedChunk],
) -> str:
    memory_block = (
        "\n\n".join(
            f"- ({idx}) [{memory.domain}/{memory.topic}] {memory.content}"
            for idx, memory in enumerate(memories, start=1)
        )
        or "- None"
    )

    doc_block = "\n\n".join(
        f"[Document {idx}] title={doc.title or 'N/A'} source_id={doc.source_id or 'N/A'} similarity={doc.similarity:.3f}\n{doc.content}"
        for idx, doc in enumerate(docs, start=1)
    )
    doc_block = truncate(doc_block, settings.max_context_chars)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant for Korean legal question answering. "
                "Always answer in Korean and prioritize the provided documents and memories. "
                "If the evidence is weak, do not sound certain and explicitly state uncertainty. "
                "Add one final line saying this is general legal information, not professional legal advice."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[Question]\n{question}\n\n"
                f"[Routing]\nsource_type={route.source_type}, topic={route.topic}, collection={route.collection}\n\n"
                f"[Relevant memories]\n{memory_block}\n\n"
                f"[Retrieved documents]\n{doc_block}\n\n"
                "Requirements:\n"
                "1. Core answer\n"
                "2. Applicable law / precedent / constitutional perspective\n"
                "3. Cautions based on the provided evidence only\n"
                "4. Put this exact sentence on the last line: "
                "'※ 본 답변은 일반적인 법률 정보 제공이며 구체적 사건은 전문가 검토가 필요할 수 있습니다.'"
            ),
        },
    ]
    return await ainvoke_text(model, messages)
