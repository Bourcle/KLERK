from utils.config import Settings
from parsers import ARTICLE_PATTERN, CASE_NUMBER_PATTERN, tokenize_koreanish
from data_structure.schemas import RouteDecision


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
    "고소",
    "고발",
    "사기",
    "횡령",
    "배임",
    "절도",
    "폭행",
    "기소",
    "무죄",
    "유죄",
}
CONSTITUTIONAL_KEYWORDS = {"헌법", "헌법재판소", "위헌", "헌재"}
PRECEDENT_KEYWORDS = {"판례", "대법원", "하급심", "선고", "사건번호"}


def select_collection(topic: str, source_type: str, settings: Settings) -> str:
    if source_type == "precedent":
        return "korean_precedent"
    if source_type == "constitutional":
        return "korean_constitutional"
    if not settings.use_topic_collections:
        return settings.default_collection
    if topic == "civil":
        return "korean_civil_law"
    if topic == "criminal":
        return "korean_criminal_law"
    return settings.default_collection


def heuristic_route(question: str, settings: Settings) -> RouteDecision:
    tokens = tokenize_koreanish(question)

    if CASE_NUMBER_PATTERN.search(question) or PRECEDENT_KEYWORDS & tokens:
        topic = "criminal" if CRIMINAL_KEYWORDS & tokens else "civil" if CIVIL_KEYWORDS & tokens else "general"
        return RouteDecision(
            source_type="precedent",
            topic=topic,
            collection=select_collection(topic, "precedent", settings),
            reason="heuristic: precedent-like pattern",
        )

    if CONSTITUTIONAL_KEYWORDS & tokens:
        return RouteDecision(
            source_type="constitutional",
            topic="general",
            collection=select_collection("general", "constitutional", settings),
            reason="heuristic: constitutional keyword",
        )

    if CRIMINAL_KEYWORDS & tokens:
        return RouteDecision(
            source_type="law",
            topic="criminal",
            collection=select_collection("criminal", "law", settings),
            reason="heuristic: criminal keyword",
        )

    if CIVIL_KEYWORDS & tokens:
        return RouteDecision(
            source_type="law",
            topic="civil",
            collection=select_collection("civil", "law", settings),
            reason="heuristic: civil keyword",
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
                    "Reply only in this JSON format: "
                    '{"source_type":"law|precedent|constitutional",'
                    '"topic":"civil|criminal|administrative|general",'
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

    source_type = payload.get("source_type", fallback.source_type)
    topic = payload.get("topic", fallback.topic)
    if source_type not in {"law", "precedent", "constitutional"}:
        source_type = fallback.source_type
    if topic not in {"civil", "criminal", "administrative", "general"}:
        topic = fallback.topic

    return RouteDecision(
        source_type=source_type,
        topic=topic,
        collection=select_collection(topic, source_type, settings),
        reason=str(payload.get("reason", fallback.reason)),
    )
