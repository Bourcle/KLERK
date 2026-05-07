from pathlib import Path

from eval.dataset import load_jsonl


def test_eval_dataset_loader_reads_jsonl(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id":"1","question":"질문","reference_answer":"답변"}\n',
        encoding="utf-8",
    )

    rows = load_jsonl(dataset)

    assert rows == [{"id": "1", "question": "질문", "reference_answer": "답변"}]
