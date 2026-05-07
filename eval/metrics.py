from statistics import mean
from typing import Any


def unavailable(name: str, reason: str) -> dict[str, Any]:
    return {
        "metric": name,
        "available": False,
        "reason": reason,
        "per_example": list(),
        "average": None,
    }


def compute_bertscore(predictions: list[str], references: list[str], *, lang: str = "ko") -> dict[str, Any]:
    """Compute BERTScore F1 between predictions and reference answers.

    Args:
        predictions: Generated answer texts to evaluate.
        references: Reference answer texts used as evaluation targets.
        lang: Language code passed to BERTScore.

    Returns:
        dict[str, Any]: Metric result containing availability, per-example scores, and average score.
    """

    try:
        from bert_score import score
    except Exception as exc:
        return unavailable("bertscore_f1", f"bert-score is not installed: {exc}")

    if not predictions:
        return unavailable("bertscore_f1", "no predictions")

    _, _, f1 = score(predictions, references, lang=lang, verbose=False)
    values = [float(item) for item in f1]
    return {
        "metric": "bertscore_f1",
        "available": True,
        "reason": "",
        "per_example": values,
        "average": mean(values) if values else None,
    }


def compute_bleurt(predictions: list[str], references: list[str], *, checkpoint: str | None = None) -> dict[str, Any]:
    """Compute BLEURT scores between predictions and reference answers.

    Args:
        predictions: Generated answer texts to evaluate.
        references: Reference answer texts used as evaluation targets.
        checkpoint: BLEURT checkpoint path used to initialize the scorer.

    Returns:
        dict[str, Any]: Metric result containing availability, per-example scores, and average score.
    """

    try:
        from bleurt import score as bleurt_score
    except Exception as exc:  # pragma: no cover - depends on optional package
        return unavailable("bleurt", f"BLEURT is not installed: {exc}")

    if not predictions:
        return unavailable("bleurt", "no predictions")
    if not checkpoint:
        return unavailable("bleurt", "BLEURT checkpoint path was not provided")

    scorer = bleurt_score.BleurtScorer(checkpoint)
    values = [float(item) for item in scorer.score(references=references, candidates=predictions)]
    return {
        "metric": "bleurt",
        "available": True,
        "reason": "",
        "per_example": values,
        "average": mean(values) if values else None,
    }


def compute_all_metrics(rows: list[dict[str, Any]], *, bleurt_checkpoint: str | None = None) -> dict[str, Any]:
    """Compute all supported reference-based evaluation metrics.

    Args:
        rows: Evaluation rows containing prediction and reference answer fields.
        bleurt_checkpoint: Optional BLEURT checkpoint path.

    Returns:
        dict[str, Any]: Metric results for BERTScore and BLEURT.
    """

    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [str(row.get("reference_answer", "")) for row in rows]
    return {
        "bertscore": compute_bertscore(predictions, references),
        "bleurt": compute_bleurt(predictions, references, checkpoint=bleurt_checkpoint),
    }
