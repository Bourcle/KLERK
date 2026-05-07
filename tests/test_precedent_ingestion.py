from build_vector_db.build_vector_db import make_precedent_document


def test_make_precedent_document_contains_metadata():
    doc, doc_id = make_precedent_document(
        {
            "case_id": "case-1",
            "title": "샘플 판례",
            "court": "대법원",
            "decision_date": "2024-01-01",
            "case_number": "2024다1",
            "summary": "요약",
            "holding": "판시사항",
            "reasoning": "이유",
            "source_url": "https://example.com",
            "tags": ["civil"],
        }
    )

    assert doc_id
    assert doc.metadata["doc_type"] == "precedent"
    assert doc.metadata["collection"] == "korean_precedent"
    assert "판시사항" in doc.page_content
