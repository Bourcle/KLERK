import hashlib
import re
from typing import Iterable

LAW_ID_PATTERN = re.compile(r"법령ID\s*:\s*([^\n]+)")
PRECEDENT_ID_PATTERN = re.compile(r"판례ID\s*:\s*([^\n]+)")
CONSTITUTIONAL_ID_PATTERN = re.compile(r"결정ID\s*:\s*([^\n]+)")
TITLE_PATTERN = re.compile(r"^##\s*(.+)$", re.MULTILINE)
CASE_NUMBER_PATTERN = re.compile(r"\d{2,4}[가-힣]{1,4}\d+")
ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조")


def stable_id(*parts: str) -> str:
    """Generate a stable short SHA-256 identifier from string parts.

    Args:
        *parts: String values to join and hash.

    Returns:
        str: First 24 characters of the SHA-256 hexadecimal digest.
    """

    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping fixed-size chunks.

    Args:
        text: Source text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        list[str]: Non-empty text chunks.

    Raises:
        ValueError: If chunk_size is not larger than overlap.
    """

    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_titles(text: str) -> list[str]:
    """Extract title values from text using the configured title pattern.

    Args:
        text: Source text containing title-like fields.

    Returns:
        list[str]: Extracted and stripped title strings.
    """

    return [title.strip() for title in TITLE_PATTERN.findall(text)]


def parse_law_ids(text: str) -> list[str]:
    """Extract Korean law IDs from text using the configured law ID pattern.

    Args:
        text: Source text containing law IDs.

    Returns:
        list[str]: Extracted and stripped law ID strings.
    """

    return [value.strip() for value in LAW_ID_PATTERN.findall(text)]


def parse_precedent_ids(text: str) -> list[str]:
    """Extract precedent IDs from text using the configured precedent ID pattern.

    Args:
        text: Source text containing precedent IDs.

    Returns:
        list[str]: Extracted and stripped precedent ID strings.
    """

    return [value.strip() for value in PRECEDENT_ID_PATTERN.findall(text)]


def parse_constitutional_ids(text: str) -> list[str]:
    """Extract constitutional decision IDs from text using the configured pattern.

    Args:
        text: Source text containing constitutional decision IDs.

    Returns:
        list[str]: Extracted and stripped constitutional decision ID strings.
    """

    return [value.strip() for value in CONSTITUTIONAL_ID_PATTERN.findall(text)]


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces.

    Args:
        text: Source text to normalize.

    Returns:
        str: Text with repeated whitespace collapsed and surrounding spaces removed.
    """

    return re.sub(r"\s+", " ", text).strip()


def tokenize_koreanish(text: str) -> set[str]:
    """Tokenize mixed Korean and alphanumeric text into normalized keyword tokens.

    Args:
        text: Source text to tokenize.

    Returns:
        set[str]: Unique lowercase tokens with length of at least two characters.
    """

    cleaned = re.sub(r"[^0-9A-Za-z가-힣_\- ]+", " ", text.lower())
    return {token for token in cleaned.split() if len(token) >= 2}


def overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    """Calculate Jaccard overlap ratio between two token collections.

    Args:
        a: First token iterable.
        b: Second token iterable.

    Returns:
        float: Intersection-over-union ratio between both token sets.
    """

    a_set = set(a)
    b_set = set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)
