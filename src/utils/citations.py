import re
from typing import Any

from data_structure.schemas import RetrievedChunk
from parsers import stable_id

CITATION_RE = re.compile(r"\[E(\d+)\]")


def attach_evidence_ids(docs: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[dict[str, Any]]]:
    """Attach stable evidence IDs to retrieved chunks and build citation metadata.

    Args:
        docs: Retrieved chunks to annotate with evidence IDs.

    Returns:
        tuple[list[RetrievedChunk], list[dict[str, Any]]]: Evidence-annotated chunks and citation metadata list.
    """

    evidence_docs: list[RetrievedChunk] = []
    evidence_list: list[dict[str, Any]] = []

    for idx, doc in enumerate(docs, start=1):
        evidence_id = f"E{idx}"
        source_id = doc.source_id or stable_id(doc.source, doc.title or "", doc.content[:120])
        metadata = dict(doc.metadata or {})
        metadata["evidence_id"] = evidence_id
        evidence_doc = doc.model_copy(update={"source_id": source_id, "metadata": metadata})
        evidence_docs.append(evidence_doc)
        evidence_list.append(
            {
                "evidence_id": evidence_id,
                "label": f"[{evidence_id}]",
                "source": evidence_doc.source,
                "source_id": source_id,
                "title": evidence_doc.title,
                "collection": evidence_doc.collection,
                "similarity": evidence_doc.similarity,
                "metadata": metadata,
            }
        )

    return evidence_docs, evidence_list


def evidence_label(doc: RetrievedChunk, fallback_idx: int) -> str:
    """Return the citation label for a retrieved evidence chunk.

    Args:
        doc: Retrieved chunk containing optional evidence metadata.
        fallback_idx: Fallback evidence index used when no evidence ID exists.

    Returns:
        str: Citation label such as "[E1]".
    """

    return f"[{doc.metadata.get('evidence_id') or f'E{fallback_idx}'}]"


def validate_citations(answer: str, evidence_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate generated answer citations against available evidence IDs.

    Args:
        answer: Generated answer text containing citation labels.
        evidence_list: Available evidence metadata containing valid evidence IDs.

    Returns:
        dict[str, Any]: Citation validation result with cited IDs, valid IDs, invalid citations, and validity status.
    """

    cited = [f"E{match}" for match in CITATION_RE.findall(answer or "")]
    valid_ids = {str(item.get("evidence_id")) for item in evidence_list}
    invalid = sorted({item for item in cited if item not in valid_ids})
    return {
        "cited_evidence_ids": cited,
        "valid_evidence_ids": sorted(valid_ids),
        "invalid_citations": invalid,
        "has_citations": bool(cited),
        "valid": not invalid,
    }
