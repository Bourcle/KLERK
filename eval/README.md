# KLERK Evaluation

This directory contains reproducible evaluation scaffolding. The sample dataset is only a schema smoke test and must not be used as a resume metric.

## Dataset

JSONL fields:

- `id`
- `question`
- `reference_answer`
- `reference_sources` optional
- `domain` optional
- `notes` optional

Place the real 50-question dataset at `eval/datasets/legal_qa_50.jsonl` using the same schema.

## Run

```bash
PYTHONPATH=src:. python -m eval.run_klerk_eval --dataset eval/datasets/legal_qa_50.sample.jsonl --output eval/outputs/klerk_predictions.jsonl
PYTHONPATH=src:. python -m eval.run_mcp_baseline --dataset eval/datasets/legal_qa_50.sample.jsonl --output eval/outputs/mcp_only_predictions.jsonl
PYTHONPATH=src:. python -m eval.report --klerk eval/outputs/klerk_predictions.jsonl --baseline eval/outputs/mcp_only_predictions.jsonl --output-dir eval/outputs
```

BERTScore and BLEURT are optional. If the packages or BLEURT checkpoint are missing, the report marks the metric unavailable instead of fabricating a score.
