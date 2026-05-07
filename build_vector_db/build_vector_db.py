import argparse
import hashlib
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from chromadb import logger
from dotenv import load_dotenv

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "필수 패키지가 없습니다. "
        "uv add langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers pypdf python-dotenv"
    ) from exc


load_dotenv()


@dataclass(frozen=True)
class LawSpec:
    law_name: str
    collection_name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Token:
    kind: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class LawChunk:
    chunk_type: str
    law_name: str
    source: str
    article_no: str | None
    article_title: str | None
    part: str | None
    chapter: str | None
    section: str | None
    subsection: str | None
    text: str


YUKBEOP_SPECS: tuple[LawSpec, ...] = (
    LawSpec("대한민국헌법", "law_constitution", ("헌법", "constitution")),
    LawSpec("민법", "law_civil", ("civil", "civil_code")),
    LawSpec("형법", "law_criminal", ("형사법", "criminal", "criminal_code")),
    LawSpec("상법", "law_commercial", ("commercial", "commercial_code")),
    LawSpec("민사소송법", "law_civil_procedure", ("민사소송", "civil_procedure", "civil procedure")),
    LawSpec("형사소송법", "law_criminal_procedure", ("형사소송", "criminal_procedure", "criminal procedure")),
)
PRECEDENT_COLLECTION_NAME = "korean_precedent"


LAW_CENTER_HEADER_RE = re.compile(r"법제처\s+\d+\s+국가법령정보센터")
PAGE_NUMBER_RE = re.compile(r"^-?\s*\d+\s*-?$")
BOOK_HEADER_RE = re.compile(r"^\s*(?:六法全書|육법전서)\s*$")
PROMULGATION_LINE_RE = re.compile(r"^\[시행[^\]]*\]\s*\[[^\]]*\]$")

PART_RE = re.compile(r"(?m)^\s*제\s*\d+\s*편\s+.+$")
CHAPTER_RE = re.compile(r"(?m)^\s*제\s*\d+\s*장\s+.+$")
SECTION_RE = re.compile(r"(?m)^\s*제\s*\d+\s*절\s+.+$")
SUBSECTION_RE = re.compile(r"(?m)^\s*제\s*\d+\s*관\s+.+$")
APPENDIX_RE = re.compile(r"(?m)^\s*부칙(?:\s*<[^>]+>|\s*\[[^\]]+\])?.*$")
ARTICLE_RE = re.compile(r"(?m)^\s*제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*\([^)]+\))?.*$")
ARTICLE_HEADER_RE = re.compile(r"^(제\s*\d+\s*조(?:\s*의\s*\d+)?)(?:\s*\(([^)]*)\))?")


def normalize_legal_text(text: str) -> str:
    """Normalize legal text for whitespace-insensitive matching.

    Args:
        text: Raw legal text to normalize.

    Returns:
        str: NFKC-normalized, whitespace-removed, lowercase text.
    """

    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", "", text)
    return text.strip().lower()


def compact(text: str) -> str:
    """Create a compact normalized representation of legal text.

    Args:
        text: Raw text to compact.

    Returns:
        str: Normalized text with whitespace removed.
    """

    return normalize_legal_text(text)


def collapse_spaces(text: str) -> str:
    """Normalize text and collapse repeated whitespace into single spaces.

    Args:
        text: Raw text to normalize.

    Returns:
        str: NFKC-normalized text with repeated whitespace collapsed.
    """

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def iter_pdf_pages(pdf_path: str | Path) -> Iterator[str]:
    """Yield page text from a PDF file one page at a time.

    Args:
        pdf_path: Path to the source PDF file.

    Yields:
        str: Extracted text content for each PDF page.
    """

    loader = PyPDFLoader(str(pdf_path))
    for page_doc in loader.lazy_load():
        yield page_doc.page_content or ""


def detection_lines(page_text: str) -> list[str]:
    """Preprocess page text into lines used for law-boundary detection.

    Args:
        page_text: Raw text extracted from one PDF page.

    Returns:
        list[str]: Cleaned non-empty lines with repeated headers and page numbers removed.
    """

    lines: list[str] = []
    for raw in (page_text or "").splitlines():
        line = collapse_spaces(raw)
        if not line:
            continue
        if LAW_CENTER_HEADER_RE.fullmatch(line):
            continue
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        if BOOK_HEADER_RE.fullmatch(line):
            continue
        lines.append(line)
    return lines


def clean_page_text(page_text: str, law_name: str | None = None) -> str:
    """Clean extracted page text for statute indexing.

    Args:
        page_text: Raw text extracted from one PDF page.
        law_name: Optional law name used to remove repeated page-level law headers.

    Returns:
        str: Cleaned page text suitable for downstream chunking and indexing.
    """

    normalized_law = compact(law_name or "")
    cleaned: list[str] = []

    for raw in (page_text or "").splitlines():
        line = collapse_spaces(raw)
        if not line:
            cleaned.append("")
            continue

        if LAW_CENTER_HEADER_RE.fullmatch(line):
            continue
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        if BOOK_HEADER_RE.fullmatch(line):
            continue

        if normalized_law and compact(line) in {normalized_law, compact(f"「{law_name}」")}:
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_title_line(line: str, spec: LawSpec) -> bool:
    """Check whether a line matches a law title or one of its aliases.

    Args:
        line: Candidate line from extracted PDF text.
        spec: Law specification containing law name and aliases.

    Returns:
        bool: True when the line exactly matches the law title or alias after normalization.
    """

    normalized_line = compact(line)
    candidates = {compact(spec.law_name), compact(f"「{spec.law_name}」")}
    for alias in spec.aliases:
        candidates.add(compact(alias))
        candidates.add(compact(f"「{alias}」"))
    return normalized_line in candidates


def start_cues_for_spec(spec: LawSpec) -> list[re.Pattern[str]]:
    """Build law-start cue patterns for a specific law specification.

    Args:
        spec: Law specification used to select common or constitution-specific start cues.

    Returns:
        list[re.Pattern[str]]: Regular expression patterns indicating the beginning of a law block.
    """

    common = [
        PROMULGATION_LINE_RE,
        re.compile(r"^제\s*1\s*편\b"),
        re.compile(r"^제\s*1\s*장\b"),
        re.compile(r"^제\s*1\s*절\b"),
        re.compile(r"^제\s*1\s*조(?:\s*의\s*\d+)?"),
        re.compile(r"^총칙$"),
        re.compile(r"^통칙$"),
        re.compile(r"^부칙"),
    ]

    if spec.collection_name == "law_constitution":
        return [
            re.compile(r"^전문$"),
            PROMULGATION_LINE_RE,
            re.compile(r"^제\s*1\s*장\b"),
            re.compile(r"^제\s*1\s*조(?:\s*의\s*\d+)?"),
        ]

    return common


def page_has_law_start_signature(
    page_texts: list[str],
    page_idx: int,
    spec: LawSpec,
    *,
    top_title_window: int = 8,
    current_page_window: int = 40,
    next_page_window: int = 15,
) -> bool:
    """Determine whether a PDF page is the start page of a specific law.

    Args:
        page_texts: Extracted text for all PDF pages.
        page_idx: Zero-based page index to inspect.
        spec: Law specification to detect.
        top_title_window: Number of top lines searched for the law title.
        current_page_window: Number of lines after the title searched for start cues.
        next_page_window: Number of lines from the next page searched for start cues.

    Returns:
        bool: True when the page contains both a law title and a valid start cue.
    """

    current_lines = detection_lines(page_texts[page_idx])
    if not current_lines:
        return False

    title_pos: int | None = None
    for idx, line in enumerate(current_lines[:top_title_window]):
        if is_title_line(line, spec):
            title_pos = idx
            break

    if title_pos is None:
        return False

    next_lines: list[str] = []
    if page_idx + 1 < len(page_texts):
        next_lines = detection_lines(page_texts[page_idx + 1])[:next_page_window]

    window = current_lines[title_pos + 1 : title_pos + 1 + current_page_window] + next_lines
    cues = start_cues_for_spec(spec)

    return any(pattern.search(line) for line in window for pattern in cues)


def detect_law_start_pages(page_texts: list[str]) -> dict[str, int]:
    """Detect one-based start pages for each six-code law in a single PDF.

    Args:
        page_texts: Extracted text for all pages in the source PDF.

    Returns:
        dict[str, int]: Mapping from collection name to one-based physical start page.

    Raises:
        ValueError: If any law start page is missing or detected out of order.
    """

    found: dict[str, int] = {}

    for page_idx in range(len(page_texts)):
        for spec in YUKBEOP_SPECS:
            if spec.collection_name in found:
                continue
            if page_has_law_start_signature(page_texts, page_idx, spec):
                found[spec.collection_name] = page_idx + 1
                break

    missing = [spec.law_name for spec in YUKBEOP_SPECS if spec.collection_name not in found]
    if missing:
        raise ValueError(f"법 시작 페이지를 찾지 못했습니다: {missing}")

    ordered = [found[spec.collection_name] for spec in YUKBEOP_SPECS]
    if ordered != sorted(ordered):
        debug = [(spec.law_name, found[spec.collection_name]) for spec in YUKBEOP_SPECS]
        raise ValueError(f"법 시작 페이지 순서가 비정상적입니다: {debug}")

    return found


def split_single_pdf_by_patterns(pdf_path: str | Path) -> dict[str, dict]:
    """Split a single six-code PDF into law-specific text blocks.

    Args:
        pdf_path: Path to the source six-code PDF.

    Returns:
        dict[str, dict]: Mapping from collection name to law metadata, page range, and extracted text.

    Raises:
        ValueError: If the PDF has no readable pages or law boundaries cannot be detected.
    """

    pdf_path = Path(pdf_path)
    page_texts = list(iter_pdf_pages(pdf_path))
    if not page_texts:
        raise ValueError("PDF에서 페이지를 읽지 못했습니다.")

    start_pages = detect_law_start_pages(page_texts)
    ordered_specs = sorted(YUKBEOP_SPECS, key=lambda spec: start_pages[spec.collection_name])

    results: dict[str, dict] = {}

    for idx, spec in enumerate(ordered_specs):
        start_page = start_pages[spec.collection_name]
        start_idx = start_page - 1

        if idx + 1 < len(ordered_specs):
            next_start_idx = start_pages[ordered_specs[idx + 1].collection_name] - 1
            end_idx_exclusive = next_start_idx
        else:
            end_idx_exclusive = len(page_texts)

        law_pages: list[str] = []
        for page_text in page_texts[start_idx:end_idx_exclusive]:
            cleaned = clean_page_text(page_text, law_name=spec.law_name)
            if cleaned:
                law_pages.append(cleaned)

        law_text = "\n\n".join(law_pages).strip()

        results[spec.collection_name] = {
            "law_name": spec.law_name,
            "collection_name": spec.collection_name,
            "start_page": start_page,
            "end_page": end_idx_exclusive,
            "text": law_text,
        }

    return results


def extract_tokens(law_text: str) -> list[Token]:
    """Extract structural statute tokens from law text.

    Args:
        law_text: Full text of one law block.

    Returns:
        list[Token]: Sorted structural tokens such as part, chapter, section, appendix, and article.
    """

    patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("part", PART_RE),
        ("chapter", CHAPTER_RE),
        ("section", SECTION_RE),
        ("subsection", SUBSECTION_RE),
        ("appendix", APPENDIX_RE),
        ("article", ARTICLE_RE),
    )

    tokens: list[Token] = []
    for kind, pattern in patterns:
        for match in pattern.finditer(law_text):
            tokens.append(Token(kind=kind, start=match.start(), end=match.end(), text=match.group(0).strip()))

    priority = {"part": 0, "chapter": 1, "section": 2, "subsection": 3, "appendix": 4, "article": 5}
    tokens.sort(key=lambda x: (x.start, priority[x.kind], -(x.end - x.start)))
    return tokens


def parse_article_header(header_line: str) -> tuple[str | None, str | None]:
    """Parse an article number and title from an article header line.

    Args:
        header_line: Raw article header line.

    Returns:
        tuple[str | None, str | None]: Article number and article title, or None values when parsing fails.
    """

    match = ARTICLE_HEADER_RE.match(collapse_spaces(header_line))
    if not match:
        return None, None
    article_no = re.sub(r"\s+", "", match.group(1))
    article_title = match.group(2).strip() if match.group(2) else None
    return article_no, article_title


def hierarchical_path(
    part: str | None,
    chapter: str | None,
    section: str | None,
    subsection: str | None,
) -> str | None:
    """Build a readable statute hierarchy path from structural headings.

    Args:
        part: Optional part heading.
        chapter: Optional chapter heading.
        section: Optional section heading.
        subsection: Optional subsection heading.

    Returns:
        str | None: Joined hierarchy path, or None when no headings exist.
    """

    items = [collapse_spaces(x) for x in (part, chapter, section, subsection) if x]
    return " > ".join(items) if items else None


def extract_constitution_preamble(law_text: str) -> tuple[str | None, str]:
    """Separate the Korean Constitution preamble from the article body.

    Args:
        law_text: Full text of the Korean Constitution or another law.

    Returns:
        tuple[str | None, str]: Preamble text when detected and remaining law text.
    """

    if compact("대한민국헌법") not in compact(law_text):
        return None, law_text

    preamble_match = re.search(r"(?m)^전문\s*$", law_text)
    first_article_match = ARTICLE_RE.search(law_text)
    if not preamble_match or not first_article_match or first_article_match.start() <= preamble_match.end():
        return None, law_text

    preamble_text = law_text[preamble_match.start() : first_article_match.start()].strip()
    remainder = law_text[first_article_match.start() :].strip()
    return preamble_text, remainder


def parse_law_chunks(
    law_text: str,
    *,
    law_name: str,
    source: str,
    include_appendix: bool = True,
) -> list[LawChunk]:
    """Parse law text into article, appendix, and optional preamble chunks.

    Args:
        law_text: Full text of one law block.
        law_name: Official law name associated with the text.
        source: Source path or identifier for provenance metadata.
        include_appendix: Whether to include appendix text as a separate chunk.

    Returns:
        list[LawChunk]: Parsed law chunks with hierarchy and article metadata.

    Raises:
        ValueError: If no structural tokens are found in the law text.
    """

    chunks: list[LawChunk] = []

    if law_name == "대한민국헌법":
        preamble_text, law_text = extract_constitution_preamble(law_text)
        if preamble_text:
            chunks.append(
                LawChunk(
                    chunk_type="preamble",
                    law_name=law_name,
                    source=source,
                    article_no=None,
                    article_title="전문",
                    part=None,
                    chapter=None,
                    section=None,
                    subsection=None,
                    text=preamble_text,
                )
            )

    tokens = extract_tokens(law_text)
    if not tokens:
        raise ValueError(f"{law_name}: 조문/편/장 토큰을 찾지 못했습니다. PDF 추출 결과를 확인해 주세요.")

    state = {"part": None, "chapter": None, "section": None, "subsection": None}

    for idx, token in enumerate(tokens):
        next_start = tokens[idx + 1].start if idx + 1 < len(tokens) else len(law_text)

        if token.kind == "part":
            state["part"] = collapse_spaces(token.text)
            state["chapter"] = None
            state["section"] = None
            state["subsection"] = None
            continue

        if token.kind == "chapter":
            state["chapter"] = collapse_spaces(token.text)
            state["section"] = None
            state["subsection"] = None
            continue

        if token.kind == "section":
            state["section"] = collapse_spaces(token.text)
            state["subsection"] = None
            continue

        if token.kind == "subsection":
            state["subsection"] = collapse_spaces(token.text)
            continue

        if token.kind == "appendix":
            if include_appendix:
                appendix_text = law_text[token.start :].strip()
                if appendix_text:
                    chunks.append(
                        LawChunk(
                            chunk_type="appendix",
                            law_name=law_name,
                            source=source,
                            article_no=None,
                            article_title="부칙",
                            part=None,
                            chapter=None,
                            section=None,
                            subsection=None,
                            text=appendix_text,
                        )
                    )
            break

        if token.kind == "article":
            article_text = law_text[token.start : next_start].strip()
            if not article_text:
                continue

            article_no, article_title = parse_article_header(token.text)
            chunks.append(
                LawChunk(
                    chunk_type="article",
                    law_name=law_name,
                    source=source,
                    article_no=article_no,
                    article_title=article_title,
                    part=state["part"],
                    chapter=state["chapter"],
                    section=state["section"],
                    subsection=state["subsection"],
                    text=article_text,
                )
            )

    return chunks


def build_summary(chunk: LawChunk, path_text: str | None) -> str:
    """Build a compact summary string for a law chunk.

    Args:
        chunk: Parsed law chunk to summarize.
        path_text: Optional hierarchy path for the chunk.

    Returns:
        str: Summary text containing law name, article metadata, and provenance fields.
    """

    parts = [chunk.law_name]
    if chunk.article_no:
        parts.append(chunk.article_no)
    if chunk.article_title:
        parts.append(f"({chunk.article_title})")
    summary_line = " ".join(parts)

    provenance_parts = [f"문서유형: 법령"]
    if chunk.article_no:
        provenance_parts.append(f"조문번호: {chunk.article_no}")
    if path_text:
        provenance_parts.append(f"위치: {path_text}")
    provenance_line = " | ".join(provenance_parts)

    return f"{summary_line}\n{provenance_line}"


def make_document(chunk: LawChunk) -> tuple[Document, str]:
    """Convert a parsed law chunk into a LangChain document and stable document ID.

    Args:
        chunk: Parsed law chunk to convert.

    Returns:
        tuple[Document, str]: LangChain document with metadata and stable document ID.
    """

    path_text = hierarchical_path(chunk.part, chunk.chapter, chunk.section, chunk.subsection)

    if chunk.chunk_type == "appendix":
        source_message = f"다음 내용은 **{chunk.law_name}**의 **부칙**에서 발췌한 내용입니다."
    elif chunk.chunk_type == "preamble":
        source_message = f"다음 내용은 **{chunk.law_name}**의 **전문**에서 발췌한 내용입니다."
    elif path_text:
        source_message = f"다음 조항은 **{chunk.law_name}**의 **{path_text}**에서 발췌한 내용입니다."
    else:
        source_message = f"다음 조항은 **{chunk.law_name}**에서 발췌한 내용입니다."

    summary_text = build_summary(chunk, path_text)

    metadata = {
        "law_name": chunk.law_name,
        "source": chunk.source,
        "chunk_type": chunk.chunk_type,
        "article_no": chunk.article_no,
        "article_title": chunk.article_title,
        "part": chunk.part,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "hierarchy": path_text,
        "doc_type": "statute",
        "summary": summary_text,
        "source_type": "law",
    }

    content = (
        f"<요약>\n{summary_text}\n</요약>\n\n"
        f"<출처>\n{source_message}\n</출처>\n\n"
        f"<법률조항>\n{chunk.text}\n</법률조항>"
    )

    stable_key = "|".join(
        [
            chunk.law_name,
            chunk.chunk_type,
            chunk.article_no or chunk.article_title or "misc",
            path_text or "",
            hashlib.sha1(chunk.text.encode("utf-8")).hexdigest(),
        ]
    )
    doc_id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
    return Document(page_content=content, metadata=metadata), doc_id


def batched(iterable: list, batch_size: int) -> Iterator[list]:
    """Yield fixed-size batches from a list.

    Args:
        iterable: Source list to split into batches.
        batch_size: Maximum number of items per batch.

    Yields:
        list: Consecutive batch slices from the source list.
    """

    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def get_embeddings(
    model_name: str = "BAAI/bge-m3",
    device: str = os.getenv("EMB_DEVICE", "cpu"),
    batch_size: int = 32,
    multi_process: bool = False,
) -> HuggingFaceEmbeddings:
    """Create a local Hugging Face embedding model for vector indexing.

    Args:
        model_name: Hugging Face model name or local cached model path.
        device: Device used for embedding inference.
        batch_size: Encoding batch size.
        multi_process: Whether to enable multi-process embedding.

    Returns:
        HuggingFaceEmbeddings: Configured Hugging Face embedding backend.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    resolved_model_name = resolve_local_hf_snapshot(model_name)

    model_kwargs = {"local_files_only": True}
    if device:
        model_kwargs["device"] = device

    print(f"Using device: {device} ; Using batch size: {batch_size} ; Using multi_process: {multi_process}", flush=True)

    return HuggingFaceEmbeddings(
        model_name=resolved_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": True,
            # 필요하면 아래도 실험 가능
            # "precision": "float32",   # 또는 "int8"
        },
        multi_process=multi_process,
        show_progress=True,
    )


def resolve_local_hf_snapshot(model_name: str) -> str:
    """Resolve a locally cached Hugging Face snapshot path when available.

    Args:
        model_name: Hugging Face model name or local model path.

    Returns:
        str: Valid local snapshot path when found, otherwise the original model name.
    """

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name
    candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)
    required_files = ("config.json", "modules.json", "tokenizer.json")
    for snapshot in candidates:
        if all((snapshot / name).exists() for name in required_files):
            return str(snapshot)
    return model_name


def find_law_spec(law_name_or_collection: str) -> LawSpec | None:
    """Find a law specification by law name, collection name, or alias.

    Args:
        law_name_or_collection: Law name, collection name, or alias to resolve.

    Returns:
        LawSpec | None: Matching law specification, or None when no match is found.
    """

    target = compact(law_name_or_collection)
    for spec in YUKBEOP_SPECS:
        for candidate in (spec.law_name, spec.collection_name, *spec.aliases):
            if compact(candidate) == target:
                return spec
    return None


def build_collections_from_single_pdf(
    pdf_path: str | Path,
    persist_dir: str | Path,
    embeddings: HuggingFaceEmbeddings,
    *,
    batch_size: int = 8,
    include_appendix: bool = True,
    dump_split_text: bool = False,
) -> list[dict]:
    """Build Chroma collections for all six-code laws from a single PDF.

    Args:
        pdf_path: Path to the source six-code PDF.
        persist_dir: Directory where Chroma collections and optional split text are stored.
        embeddings: Embedding backend used by Chroma.
        batch_size: Number of documents inserted per Chroma batch.
        include_appendix: Whether to include appendix chunks.
        dump_split_text: Whether to save split law text files for debugging.

    Returns:
        list[dict]: Build reports for each generated law collection.
    """

    pdf_path = Path(pdf_path)
    persist_dir = Path(persist_dir)

    split_map = split_single_pdf_by_patterns(pdf_path)

    reports: list[dict] = []
    split_dir = persist_dir / "_split_text"
    if dump_split_text:
        split_dir.mkdir(parents=True, exist_ok=True)

    for spec in YUKBEOP_SPECS:
        block = split_map[spec.collection_name]
        law_text = block["text"]

        print(f"[START] {pdf_path.name} -> {spec.collection_name}")

        if dump_split_text:
            split_path = split_dir / f"{spec.collection_name}.txt"
            split_path.write_text(law_text, encoding="utf-8")

        chunks = parse_law_chunks(
            law_text,
            law_name=spec.law_name,
            source=str(pdf_path),
            include_appendix=include_appendix,
        )

        documents: list[Document] = []
        ids: list[str] = []
        for chunk in chunks:
            doc, doc_id = make_document(chunk)
            documents.append(doc)
            ids.append(doc_id)

        vector_store = Chroma(
            collection_name=spec.collection_name,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

        for doc_batch, id_batch in zip(batched(documents, batch_size), batched(ids, batch_size)):
            vector_store.add_documents(documents=doc_batch, ids=id_batch)

        report = {
            "law_name": spec.law_name,
            "collection_name": spec.collection_name,
            "pdf_path": str(pdf_path),
            "start_page": block["start_page"],
            "end_page": block["end_page"],
            "num_chunks": len(documents),
            "num_article_chunks": sum(1 for x in chunks if x.chunk_type == "article"),
            "num_appendix_chunks": sum(1 for x in chunks if x.chunk_type == "appendix"),
            "num_preamble_chunks": sum(1 for x in chunks if x.chunk_type == "preamble"),
        }
        reports.append(report)

        print(
            f"[DONE] {spec.law_name}: pages={block['start_page']}..{block['end_page']} "
            f"chunks={report['num_chunks']} "
            f"(articles={report['num_article_chunks']}, "
            f"preamble={report['num_preamble_chunks']}, appendix={report['num_appendix_chunks']})"
        )

    return reports


def build_all(
    *,
    pdf_path: str | Path,
    persist_dir: str | Path,
    embedding_model: str = "BAAI/bge-m3",
    batch_size: int = 8,
    include_appendix: bool = True,
    reset: bool = True,
    dump_split_text: bool = False,
) -> list[dict]:
    """Build all six-code Chroma collections and write a build report.

    Args:
        pdf_path: Path to the source six-code PDF.
        persist_dir: Directory where Chroma collections and reports are stored.
        embedding_model: Hugging Face embedding model name or local path.
        batch_size: Number of documents inserted per Chroma batch.
        include_appendix: Whether to include appendix chunks.
        reset: Whether to remove the existing persist directory before building.
        dump_split_text: Whether to save split law text files for debugging.

    Returns:
        list[dict]: Build reports for all generated law collections.
    """

    persist_dir = Path(persist_dir)
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(
        model_name=embedding_model, device=os.getenv("EMB_DEVICE", "cpu"), batch_size=batch_size, multi_process=True
    )
    reports = build_collections_from_single_pdf(
        pdf_path=pdf_path,
        persist_dir=persist_dir,
        embeddings=embeddings,
        batch_size=batch_size,
        include_appendix=include_appendix,
        dump_split_text=dump_split_text,
    )

    report_path = persist_dir / "build_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return reports


def open_collection(
    law_name_or_collection: str,
    persist_dir: str | Path,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Open a Chroma collection by law name, collection name, or alias.

    Args:
        law_name_or_collection: Law name, collection name, or alias to open.
        persist_dir: Directory where Chroma collections are persisted.
        embeddings: Optional embedding backend; created automatically when omitted.

    Returns:
        Chroma: Opened Chroma collection.

    Raises:
        ValueError: If the requested law or collection is not supported.
    """

    embeddings = embeddings or get_embeddings()
    spec = find_law_spec(law_name_or_collection)
    if spec is None:
        raise ValueError(f"지원하지 않는 법/컬렉션입니다: {law_name_or_collection}")

    return Chroma(
        collection_name=spec.collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def search_collection(
    law_name_or_collection: str,
    query: str,
    persist_dir: str | Path,
    *,
    k: int = 4,
    embedding_model: str = "BAAI/bge-m3",
):
    """Search a law-specific Chroma collection.

    Args:
        law_name_or_collection: Law name, collection name, or alias to search.
        query: Search query text.
        persist_dir: Directory where Chroma collections are persisted.
        k: Number of documents to retrieve.
        embedding_model: Hugging Face embedding model name or local path.

    Returns:
        list[Document]: Retrieved documents from the selected Chroma collection.
    """

    store = open_collection(
        law_name_or_collection=law_name_or_collection,
        persist_dir=persist_dir,
        embeddings=get_embeddings(embedding_model),
    )
    return store.similarity_search(query, k=k)


def make_precedent_document(item: dict) -> tuple[Document, str]:
    """Convert a precedent JSON object into a LangChain document and stable document ID.

    Args:
        item: Precedent record containing case metadata, summary, holding, and reasoning.

    Returns:
        tuple[Document, str]: LangChain document with precedent metadata and stable document ID.
    """

    case_id = str(item.get("case_id") or item.get("case_number") or "").strip()
    title = str(item.get("title") or case_id or "Untitled precedent").strip()
    court = str(item.get("court") or "").strip()
    decision_date = str(item.get("decision_date") or "").strip()
    case_number = str(item.get("case_number") or "").strip()
    summary = str(item.get("summary") or "").strip()
    holding = str(item.get("holding") or "").strip()
    reasoning = str(item.get("reasoning") or "").strip()
    source_url = str(item.get("source_url") or "").strip()
    tags = item.get("tags") or []

    content = (
        f"<요약>\n{title}\n문서유형: 판례 | 법원: {court or 'N/A'} | "
        f"선고일: {decision_date or 'N/A'} | 사건번호: {case_number or 'N/A'}\n</요약>\n\n"
        f"<판시사항>\n{holding or summary}\n</판시사항>\n\n"
        f"<이유>\n{reasoning}\n</이유>\n\n"
        f"<출처>\n{source_url or 'N/A'}\n</출처>"
    )
    metadata = {
        "doc_type": "precedent",
        "source_type": "precedent",
        "case_id": case_id,
        "title": title,
        "court": court,
        "decision_date": decision_date,
        "case_number": case_number,
        "source_url": source_url,
        "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
        "collection": PRECEDENT_COLLECTION_NAME,
    }
    stable_key = "|".join(
        [case_id, title, court, decision_date, case_number, hashlib.sha1(content.encode("utf-8")).hexdigest()]
    )
    return Document(page_content=content, metadata=metadata), hashlib.sha1(stable_key.encode("utf-8")).hexdigest()


def build_precedent_collection_from_jsonl(
    jsonl_path: str | Path,
    persist_dir: str | Path,
    embeddings: HuggingFaceEmbeddings,
    *,
    batch_size: int = 8,
    reset_collection: bool = False,
) -> dict:
    """Build a Chroma precedent collection from a JSONL file.

    Args:
        jsonl_path: Path to the precedent JSONL file.
        persist_dir: Directory where the Chroma collection and report are stored.
        embeddings: Embedding backend used by Chroma.
        batch_size: Number of precedent documents inserted per Chroma batch.
        reset_collection: Whether to delete matching existing document IDs before insertion.

    Returns:
        dict: Build report containing collection name, source path, and document counts.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If the JSONL file is empty or required fields are missing.
    """

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"판례 JSONL을 찾지 못했습니다: {path}")

    documents: list[Document] = []
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            required = {"case_id", "title", "court", "decision_date", "case_number", "summary", "holding", "reasoning"}
            missing = required - set(item)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            doc, doc_id = make_precedent_document(item)
            documents.append(doc)
            ids.append(doc_id)

    if not documents:
        raise ValueError(f"판례 JSONL이 비어 있습니다: {path}")

    vector_store = Chroma(
        collection_name=PRECEDENT_COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    if reset_collection:
        try:
            vector_store.delete(ids=ids)
        except Exception:
            pass
    for doc_batch, id_batch in zip(batched(documents, batch_size), batched(ids, batch_size)):
        vector_store.add_documents(documents=doc_batch, ids=id_batch)

    report = {
        "collection_name": PRECEDENT_COLLECTION_NAME,
        "jsonl_path": str(path),
        "num_chunks": len(documents),
        "num_precedents": len(documents),
    }
    report_path = Path(persist_dir) / "precedent_build_report.json"
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def resolve_pdf_path(pdf_path: str | None, pdf_dir: str | None, glob_pattern: str) -> Path:
    """Resolve the single source PDF path from an explicit path or directory glob.

    Args:
        pdf_path: Explicit PDF file path.
        pdf_dir: Directory expected to contain exactly one matching PDF.
        glob_pattern: Glob pattern used when resolving from pdf_dir.

    Returns:
        Path: Resolved PDF file path.

    Raises:
        FileNotFoundError: If no matching PDF is found.
        ValueError: If neither source option is provided or multiple PDFs match.
    """

    if pdf_path:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF를 찾지 못했습니다: {path}")
        return path

    if not pdf_dir:
        raise ValueError("--pdf-path 또는 --pdf-dir 중 하나는 필요합니다.")

    candidates = sorted(Path(pdf_dir).glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"PDF를 찾지 못했습니다: {Path(pdf_dir) / glob_pattern}")
    if len(candidates) > 1:
        names = ", ".join(x.name for x in candidates[:5])
        raise ValueError(
            "--pdf-dir 안에 PDF가 여러 개 있습니다. 단일 육법전서 PDF만 대상으로 해야 합니다. "
            f"--pdf-path로 명시해 주세요. candidates={names}"
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="단일 육법전서 PDF를 6개 collection으로 분리 인덱싱합니다.")
    parser.add_argument("--pdf-path", default=None, help="육법전서.pdf 파일 경로")
    parser.add_argument("--pdf-dir", default=None, help="PDF가 1개만 들어있는 디렉토리")
    parser.add_argument("--glob", default="*.pdf", help="--pdf-dir 사용 시 PDF 검색 glob 패턴")
    parser.add_argument("--persist-dir", default="./chroma_yukbeop", help="Chroma 저장 디렉토리")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="Hugging Face embedding model")
    parser.add_argument("--batch-size", type=int, default=8, help="Chroma add_documents 배치 크기")
    parser.add_argument("--exclude-appendix", action="store_true", help="부칙 청크를 인덱싱하지 않음")
    parser.add_argument("--keep-existing", action="store_true", help="기존 persist 디렉토리를 삭제하지 않음")
    parser.add_argument("--dump-split-text", action="store_true", help="법별 분리 결과를 txt로 저장")
    parser.add_argument("--smoke-test-query", default=None, help="인덱싱 후 예시 질의")
    parser.add_argument("--smoke-test-law", default="민법", help="예시 질의를 던질 컬렉션/법 이름")
    parser.add_argument("--precedent-jsonl", default=None, help="판례 JSONL 파일 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.precedent_jsonl:
        embeddings = get_embeddings(args.embedding_model)
        report = build_precedent_collection_from_jsonl(
            jsonl_path=args.precedent_jsonl,
            persist_dir=args.persist_dir,
            embeddings=embeddings,
            batch_size=args.batch_size,
            reset_collection=not args.keep_existing,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    pdf_path = resolve_pdf_path(args.pdf_path, args.pdf_dir, args.glob)

    reports = build_all(
        pdf_path=pdf_path,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        include_appendix=not args.exclude_appendix,
        reset=not args.keep_existing,
        dump_split_text=args.dump_split_text,
    )

    print("\n[SUMMARY]")
    print(json.dumps(reports, ensure_ascii=False, indent=2))

    if args.smoke_test_query:
        print("\n[SMOKE TEST]")
        docs = search_collection(
            law_name_or_collection=args.smoke_test_law,
            query=args.smoke_test_query,
            persist_dir=args.persist_dir,
            k=3,
            embedding_model=args.embedding_model,
        )
        for idx, doc in enumerate(docs, start=1):
            preview = doc.page_content[:400].replace("\n", " ")
            print(f"{idx}. {preview}")
            print(f"   metadata={doc.metadata}")


if __name__ == "__main__":
    main()
