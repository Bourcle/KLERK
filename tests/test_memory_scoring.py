from pathlib import Path

from storages.memory_store import MemoryRepository
from data_structure.schemas import RouteDecision


def test_memory_record_and_search(tmp_path: Path):
    repo = MemoryRepository(tmp_path / "memory.sqlite", half_life_days=30, default_importance=0.5)
    repo.record_turn(
        user_id="u1",
        question="민법상 손해배상 책임이 성립하는 요건은?",
        route=RouteDecision(source_type="law", topic="civil", collection="law_civil"),
        answer="불법행위 요건과 손해 발생, 인과관계가 중요하다.",
    )

    results = repo.search(user_id="u1", query="손해배상 요건 다시 설명해줘", domain="law", top_k=3)
    assert len(results) == 1
    assert results[0].score > 0
    assert "손해배상" in results[0].content
