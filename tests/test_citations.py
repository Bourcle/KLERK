from data_structure.schemas import RetrievedChunk
from utils.citations import attach_evidence_ids, validate_citations


def test_attach_evidence_ids_is_stable_and_validates_answer_citations():
    docs = [
        RetrievedChunk(content="민법 제750조", source="vector_db", title="민법", source_id="civil-750"),
        RetrievedChunk(content="불법행위 설명", source="mcp_augmentation", title="MCP", source_id="mcp-1"),
    ]

    updated, evidence = attach_evidence_ids(docs)

    assert [item["evidence_id"] for item in evidence] == ["E1", "E2"]
    assert updated[0].metadata["evidence_id"] == "E1"
    assert validate_citations("근거는 [E1]입니다.", evidence)["valid"] is True
    invalid = validate_citations("근거는 [E3]입니다.", evidence)
    assert invalid["valid"] is False
    assert invalid["invalid_citations"] == ["E3"]
