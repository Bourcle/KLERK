from legal_memory_agent.config import Settings
from legal_memory_agent.router import heuristic_route


def test_heuristic_route_precedent_case_number():
    settings = Settings()
    route = heuristic_route("대법원 2020다12345 판례의 취지를 설명해줘", settings)
    assert route.source_type == "precedent"


def test_heuristic_route_criminal():
    settings = Settings()
    route = heuristic_route("형사 사기죄 성립 요건이 뭐야?", settings)
    assert route.source_type == "law"
    assert route.topic == "criminal"
    assert route.collection == "korean_criminal_law"


def test_heuristic_route_civil():
    settings = Settings()
    route = heuristic_route("민법상 손해배상 청구 요건 알려줘", settings)
    assert route.source_type == "law"
    assert route.topic == "civil"
    assert route.collection == "korean_civil_law"
