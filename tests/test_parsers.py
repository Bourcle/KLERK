from parsers import (
    chunk_text,
    parse_constitutional_ids,
    parse_law_ids,
    parse_precedent_ids,
)


def test_parse_law_ids():
    text = "## 민법\n법령ID: 12345\n## 형법\n법령ID: 67890"
    assert parse_law_ids(text) == ["12345", "67890"]


def test_parse_precedent_ids():
    text = "## 대법원 2020다12345\n판례ID: PREC-1"
    assert parse_precedent_ids(text) == ["PREC-1"]


def test_parse_constitutional_ids():
    text = "## 헌재 결정\n결정ID: CC-2024-1"
    assert parse_constitutional_ids(text) == ["CC-2024-1"]


def test_chunk_text():
    text = "가" * 1800
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 800
    assert len(chunks[1]) >= 700
