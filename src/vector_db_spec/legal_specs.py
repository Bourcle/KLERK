from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class LawSpec:
    law_name: str
    collection_name: str
    aliases: tuple[str, ...]
    topic: str


def normalize_legal_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).strip().lower()


YUKBEOP_SPECS: tuple[LawSpec, ...] = (
    LawSpec(
        law_name="대한민국헌법",
        collection_name="law_constitution",
        aliases=("대한민국헌법", "헌법", "constitution"),
        topic="constitution",
    ),
    LawSpec(
        law_name="민법",
        collection_name="law_civil",
        aliases=("민법", "civil", "civil_code"),
        topic="civil",
    ),
    LawSpec(
        law_name="형법",
        collection_name="law_criminal",
        aliases=("형법", "criminal", "criminal_code"),
        topic="criminal",
    ),
    LawSpec(
        law_name="상법",
        collection_name="law_commercial",
        aliases=("상법", "commercial", "commercial_code"),
        topic="commercial",
    ),
    LawSpec(
        law_name="민사소송법",
        collection_name="law_civil_procedure",
        aliases=("민사소송법", "민사소송", "civil_procedure", "civil procedure"),
        topic="civil_procedure",
    ),
    LawSpec(
        law_name="형사소송법",
        collection_name="law_criminal_procedure",
        aliases=("형사소송법", "형사소송", "criminal_procedure", "criminal procedure"),
        topic="criminal_procedure",
    ),
)

PROCEDURE_FIRST_COLLECTION_ORDER: tuple[str, ...] = (
    "law_civil_procedure",
    "law_criminal_procedure",
    "law_constitution",
    "law_civil",
    "law_criminal",
    "law_commercial",
)

YUKBEOP_SPEC_BY_COLLECTION = {spec.collection_name: spec for spec in YUKBEOP_SPECS}
YUKBEOP_SPEC_BY_TOPIC = {spec.topic: spec for spec in YUKBEOP_SPECS}


def ordered_yukbeop_specs() -> tuple[LawSpec, ...]:
    return tuple(YUKBEOP_SPEC_BY_COLLECTION[name] for name in PROCEDURE_FIRST_COLLECTION_ORDER)


def collection_name_from_law_spec(spec: LawSpec) -> str:
    return spec.collection_name


def find_law_spec(value: str) -> LawSpec | None:
    normalized = normalize_legal_text(value)
    for spec in ordered_yukbeop_specs():
        if normalized == normalize_legal_text(spec.law_name):
            return spec
        if normalized == normalize_legal_text(spec.collection_name):
            return spec
        if any(normalized == normalize_legal_text(alias) for alias in spec.aliases):
            return spec
    return None


def match_law_spec_from_text(text: str) -> LawSpec | None:
    normalized = normalize_legal_text(text)
    matches: list[tuple[int, int, LawSpec]] = []
    for priority, spec in enumerate(ordered_yukbeop_specs()):
        alias_lengths = [
            len(normalize_legal_text(alias))
            for alias in (spec.law_name, spec.collection_name, *spec.aliases)
            if normalize_legal_text(alias) and normalize_legal_text(alias) in normalized
        ]
        if alias_lengths:
            matches.append((max(alias_lengths), -priority, spec))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][2]
