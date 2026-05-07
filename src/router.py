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
    """Normalize a topic name and resolve known aliases.

    Args:
        topic: Raw topic name from user input, model output, or route metadata.

    Returns:
        str: Normalized topic name, or "general" when no alias is matched.
    """

    normalized = (topic or "").strip().lower()
    return TOPIC_ALIASES.get(normalized, "general")


def collection_name_from_topic(topic: str) -> str | None:
    """Resolve a vector collection name from a legal topic.

    Args:
        topic: Legal topic name to resolve.

    Returns:
        str | None: Matching collection name, or None when the topic has no mapped law spec.
    """

    spec = YUKBEOP_SPEC_BY_TOPIC.get(normalize_topic(topic))
    if spec is None:
        return None
    return collection_name_from_law_spec(spec)


def resolve_law_spec_from_query(question: str):
    """Resolve a law specification directly mentioned in the question.

    Args:
        question: User question that may contain a law name, collection name, or alias.

    Returns:
        LawSpec | None: Matching law specification, or None when no law is detected.
    """

    return match_law_spec_from_text(question)


def infer_topic_from_query(question: str) -> str:
    """Infer the legal topic from law aliases, keywords, and token rules.

    Args:
        question: User question to classify into a legal topic.

    Returns:
        str: Inferred topic name, or "general" when no specific topic is detected.
    """

    spec = resolve_law_spec_from_query(question)
    if spec is not None:
        return spec.topic

    tokens = tokenize_koreanish(question)
    for topic, keywords in TOPIC_KEYWORD_RULES:
        if keywords & tokens:
            return topic
        if any(keyword in question for keyword in keywords):
            return topic
    return "general"


def select_collection(topic: str, source_type: str, settings: Settings) -> str:
    """Select the vector collection for a topic and legal source type.

    Args:
        topic: Legal topic used for collection routing.
        source_type: Legal source type such as law, precedent, or constitutional.
        settings: Runtime settings containing default and topic-collection options.

    Returns:
        str: Selected vector collection name.
    """

    if source_type == "precedent":
        return "korean_precedent"

    normalized_topic = normalize_topic(topic)

    if source_type == "constitutional":
        return collection_name_from_topic("constitution") or settings.default_collection

    if not settings.use_topic_collections:
        return settings.default_collection

    return collection_name_from_topic(normalized_topic) or settings.default_collection


def heuristic_route(question: str, settings: Settings) -> RouteDecision:
    """Create a deterministic route decision from legal patterns and keyword rules.

    Args:
        question: User question to route.
        settings: Runtime settings used for collection selection.

    Returns:
        RouteDecision: Heuristic route containing source type, topic, collection, and reason.
    """

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


async def rewrite_query_for_retrieval(question: str, route: RouteDecision, model) -> str:
    """Rewrite a legal question into a retrieval-optimized Korean search query.

    Args:
        question: Original user question.
        route: Route decision containing source type and topic context.
        model: LLM client or runnable used for query rewriting.

    Returns:
        str: Rewritten retrieval query, or the original question when rewriting fails.
    """

    from llm_model.llm import ainvoke_json

    payload = await ainvoke_json(
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Korean legal search query optimizer. "
                    "Rewrite the user question into a concise, keyword-rich Korean search query "
                    "optimized for vector similarity search in a Korean legal database.\n"
                    "Rules:\n"
                    "- Expand abbreviations (민소→민사소송법, 형소→형사소송법, 헌재→헌법재판소, etc.)\n"
                    "- Include full legal term names alongside common terms\n"
                    "- Keep article numbers (제N조) if mentioned\n"
                    "- Add relevant legal concepts related to the question\n"
                    "- Remove conversational filler; keep only substantive terms\n"
                    "- Output must be in Korean\n"
                    'Reply only in JSON: {"rewritten_query": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (f"Question: {question}\n" f"Domain: {route.source_type}, Topic: {route.topic}"),
            },
        ],
        default={"rewritten_query": question},
    )
    return str(payload.get("rewritten_query", question)).strip() or question


async def refine_query_for_retry(
    question: str, current_query: str, route: RouteDecision, doc_summaries: str, model
) -> str:
    """Refine a failed retrieval query for the next retry attempt.

    Args:
        question: Original user question.
        current_query: Previous retrieval query that produced insufficient results.
        route: Route decision containing source type and topic context.
        doc_summaries: Preview of currently retrieved documents.
        model: LLM client or runnable used for query refinement.

    Returns:
        str: Refined retrieval query, or the original question when refinement fails.
    """

    from llm_model.llm import ainvoke_json

    payload = await ainvoke_json(
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Korean legal search query optimizer. "
                    "The previous search query returned insufficient results. "
                    "Generate a refined search query using different terms, broader or narrower scope, "
                    "or alternative legal concepts.\n"
                    "- Try synonyms, related legal terms, or different perspectives\n"
                    "- If the question is about a specific article, try searching for the parent law\n"
                    "- If too specific, broaden; if too broad, narrow down\n"
                    "- Output must be in Korean\n"
                    'Reply only in JSON: {"refined_query": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n"
                    f"Previous query: {current_query}\n"
                    f"Domain: {route.source_type}, Topic: {route.topic}\n"
                    f"Current results preview:\n{doc_summaries}"
                ),
            },
        ],
        default={"refined_query": question},
    )
    return str(payload.get("refined_query", question)).strip() or question


async def route_question(question: str, model, settings: Settings) -> RouteDecision:
    """Route a Korean legal question using heuristic fallback and LLM classification.

    Args:
        question: User question to route.
        model: LLM client or runnable used for route classification.
        settings: Runtime settings used for fallback routing and collection selection.

    Returns:
        RouteDecision: Final route containing source type, topic, collection, and reason.
    """

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
