from utils.config import Settings
from parsers import ARTICLE_PATTERN, CASE_NUMBER_PATTERN, tokenize_koreanish
from data_structure.schemas import RouteDecision
from vector_db_spec.legal_specs import (
    YUKBEOP_SPEC_BY_TOPIC,
    collection_name_from_law_spec,
    match_law_spec_from_text,
)


CONSTITUTION_KEYWORDS = {
    "헌법",
    "대한민국헌법",
    "헌법재판소",
    "헌재",
    "위헌",
    "평등권",
    "기본권",
}
CIVIL_KEYWORDS = {
    "민법",
    "계약",
    "손해배상",
    "채권",
    "채무",
    "임대차",
    "부동산",
    "상속",
    "이혼",
    "위자료",
}
CRIMINAL_KEYWORDS = {
    "형법",
    "형사",
    "사기",
    "횡령",
    "배임",
    "절도",
    "폭행",
    "무죄",
    "유죄",
}
COMMERCIAL_KEYWORDS = {
    "상법",
    "주식회사",
    "주주총회",
    "이사",
    "감사",
    "상행위",
    "어음",
    "보험",
    "해상",
}
CIVIL_PROCEDURE_KEYWORDS = {
    "민사소송법",
    "민사소송",
    "소장",
    "변론",
    "항소",
    "재심",
    "관할",
    "집행",
    "가압류",
    "가처분",
}
CRIMINAL_PROCEDURE_KEYWORDS = {
    "형사소송법",
    "형사소송",
    "구속",
    "압수수색",
    "영장",
    "증거능력",
    "공소",
    "불기소",
    "체포",
    "기소",
}
PRECEDENT_KEYWORDS = {"판례", "대법원", "하급심", "선고", "사건번호"}

TOPIC_ALIASES = {
    "constitutional": "constitution",
    "constitution": "constitution",
    "law_constitution": "constitution",
    "civil": "civil",
    "law_civil": "civil",
    "criminal": "criminal",
    "law_criminal": "criminal",
    "commercial": "commercial",
    "law_commercial": "commercial",
    "civil_procedure": "civil_procedure",
    "civil procedure": "civil_procedure",
    "law_civil_procedure": "civil_procedure",
    "criminal_procedure": "criminal_procedure",
    "criminal procedure": "criminal_procedure",
    "law_criminal_procedure": "criminal_procedure",
    "general": "general",
    "administrative": "general",
}

TOPIC_KEYWORD_RULES = (
    ("civil_procedure", CIVIL_PROCEDURE_KEYWORDS),
    ("criminal_procedure", CRIMINAL_PROCEDURE_KEYWORDS),
    ("constitution", CONSTITUTION_KEYWORDS),
    ("commercial", COMMERCIAL_KEYWORDS),
    ("criminal", CRIMINAL_KEYWORDS),
    ("civil", CIVIL_KEYWORDS),
)


def normalize_topic(topic: str | None) -> str:
    normalized = (topic or "").strip().lower()
    return TOPIC_ALIASES.get(normalized, "general")


def collection_name_from_topic(topic: str) -> str | None:
    spec = YUKBEOP_SPEC_BY_TOPIC.get(normalize_topic(topic))
    if spec is None:
        return None
    return collection_name_from_law_spec(spec)


def resolve_law_spec_from_query(question: str):
    return match_law_spec_from_text(question)


def infer_topic_from_query(question: str) -> str:
    spec = resolve_law_spec_from_query(question)
    if spec is not None:
        return spec.topic

    tokens = tokenize_koreanish(question)
    for topic, keywords in TOPIC_KEYWORD_RULES:
        if keywords & tokens:
            return topic
    return "general"


def select_collection(topic: str, source_type: str, settings: Settings) -> str:
    if source_type == "precedent":
        return "korean_precedent"

    normalized_topic = normalize_topic(topic)

    if source_type == "constitutional":
        return collection_name_from_topic("constitution") or settings.default_collection

    if not settings.use_topic_collections:
        return settings.default_collection

    return collection_name_from_topic(normalized_topic) or settings.default_collection


def heuristic_route(question: str, settings: Settings) -> RouteDecision:
    inferred_topic = infer_topic_from_query(question)
    inferred_source_type = "constitutional" if inferred_topic == "constitution" else "law"
    tokens = tokenize_koreanish(question)

    if CASE_NUMBER_PATTERN.search(question) or PRECEDENT_KEYWORDS & tokens:
        topic = inferred_topic if inferred_topic != "constitution" else "general"
        return RouteDecision(
            source_type="precedent",
            topic=topic,
            collection=select_collection(topic, "precedent", settings),
            reason="heuristic: precedent-like pattern",
        )

    if inferred_source_type == "constitutional":
        return RouteDecision(
            source_type="constitutional",
            topic="constitution",
            collection=select_collection("constitution", "constitutional", settings),
            reason="heuristic: constitution alias/keyword",
        )

    if inferred_topic != "general":
        return RouteDecision(
            source_type="law",
            topic=inferred_topic,
            collection=select_collection(inferred_topic, "law", settings),
            reason="heuristic: yukbeop alias/keyword",
        )

    if ARTICLE_PATTERN.search(question):
        return RouteDecision(
            source_type="law",
            topic="general",
            collection=select_collection("general", "law", settings),
            reason="heuristic: article reference",
        )

    return RouteDecision(
        source_type="law",
        topic="general",
        collection=select_collection("general", "law", settings),
        reason="heuristic: default",
    )


async def route_question(question: str, model, settings: Settings) -> RouteDecision:
    from llm_model.llm import ainvoke_json

    fallback = heuristic_route(question, settings)
    payload = await ainvoke_json(
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a router for Korean legal queries. "
                    "Prioritize exact law names or aliases over generic keywords. "
                    "Give civil/criminal procedure priority over civil/criminal substantive law when both appear. "
                    "Reply only in JSON format: "
                    '{"source_type":"law|precedent|constitutional",'
                    '"topic":"constitution|civil|criminal|commercial|civil_procedure|criminal_procedure|general",'
                    '"reason":"..."}'
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}",
            },
        ],
        default=fallback.model_dump(),
    )

    source_type = str(payload.get("source_type", fallback.source_type))
    topic = normalize_topic(payload.get("topic", fallback.topic))

    if source_type not in {"law", "precedent", "constitutional"}:
        source_type = fallback.source_type

    resolved_spec = resolve_law_spec_from_query(question)
    if resolved_spec is not None:
        topic = resolved_spec.topic
        if topic == "constitution" and source_type != "precedent":
            source_type = "constitutional"

    if source_type == "constitutional":
        topic = "constitution"
    if source_type == "precedent" and topic == "constitution":
        topic = "general"

    return RouteDecision(
        source_type=source_type,
        topic=topic,
        collection=select_collection(topic, source_type, settings),
        reason=str(payload.get("reason", fallback.reason)),
    )
