from utils.config import Settings
from router import heuristic_route, select_collection


def test_heuristic_route_constitution():
    settings = Settings()
    route = heuristic_route("헌법 제11조 평등권", settings)
    assert route.source_type == "constitutional"
    assert route.topic == "constitution"
    assert route.collection == "law_constitution"


def test_heuristic_route_civil():
    settings = Settings()
    route = heuristic_route("임대차 계약 해지와 손해배상", settings)
    assert route.source_type == "law"
    assert route.topic == "civil"
    assert route.collection == "law_civil"


def test_heuristic_route_criminal():
    settings = Settings()
    route = heuristic_route("사기죄와 횡령죄 차이", settings)
    assert route.source_type == "law"
    assert route.topic == "criminal"
    assert route.collection == "law_criminal"


def test_heuristic_route_commercial():
    settings = Settings()
    route = heuristic_route("주주총회 결의 취소", settings)
    assert route.source_type == "law"
    assert route.topic == "commercial"
    assert route.collection == "law_commercial"


def test_heuristic_route_civil_procedure_priority():
    settings = Settings()
    route = heuristic_route("민사소송 항소기간과 관할", settings)
    assert route.source_type == "law"
    assert route.topic == "civil_procedure"
    assert route.collection == "law_civil_procedure"


def test_heuristic_route_criminal_procedure_priority():
    settings = Settings()
    route = heuristic_route("구속영장 청구와 증거능력", settings)
    assert route.source_type == "law"
    assert route.topic == "criminal_procedure"
    assert route.collection == "law_criminal_procedure"


def test_heuristic_route_precedent_keeps_collection():
    settings = Settings()
    route = heuristic_route("대법원 판례로 본 사기죄", settings)
    assert route.source_type == "precedent"
    assert route.collection == "korean_precedent"


def test_select_collection_falls_back_to_default_when_topic_collections_disabled():
    settings = Settings(use_topic_collections=False, default_collection="korean_law")
    assert select_collection("civil", "law", settings) == "korean_law"
    assert select_collection("constitution", "constitutional", settings) == "law_constitution"
