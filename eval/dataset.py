import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"id", "question", "reference_answer"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL dataset.

    Args:
        path: Path to the JSONL dataset file.

    Returns:
        list[dict[str, Any]]: Loaded dataset rows.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a row is missing required fields or the dataset is empty.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = REQUIRED_FIELDS - set(item)
            if missing:
                raise ValueError(f"{dataset_path}:{line_no} missing fields: {sorted(missing)}")
            rows.append(item)

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL file.

    Args:
        path: Output JSONL file path.
        rows: Rows to serialize as JSONL records.

    Returns:
        None: This function completes after writing all rows.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
