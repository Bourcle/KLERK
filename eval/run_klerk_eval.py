import argparse
import asyncio

from dotenv import load_dotenv

from eval.dataset import load_jsonl, write_jsonl
from src.service import LegalAgentService
from src.utils.config import get_settings


async def run(dataset_path: str, output_path: str) -> None:
    """Run KLERK evaluation over a JSONL dataset and write prediction outputs.

    Args:
        dataset_path: Path to the input evaluation JSONL dataset.
        output_path: Path where generated evaluation outputs should be written.

    Returns:
        None: This function completes after writing all prediction rows.
    """

    load_dotenv(override=True)
    settings = get_settings()
    service = LegalAgentService(settings)
    rows = load_jsonl(dataset_path)
    outputs = []

    try:
        for item in rows:
            result = await service.aask(
                question=item["question"],
                user_id=str(item.get("user_id", "eval-user")),
                thread_id=f"eval-{item['id']}",
            )
            outputs.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "reference_sources": item.get("reference_sources", []),
                    "prediction": result.answer,
                    "route": result.route.model_dump(mode="json"),
                    "selected_domain": result.route.topic,
                    "selected_collection": result.route.collection,
                    "retrieval_sufficient": result.retrieval_sufficient,
                    "sufficiency_reason": result.sufficiency_reason,
                    "used_mcp": result.used_mcp,
                    "retrieval_iterations": result.retrieval_iterations,
                    "fallback_history": result.fallback_history,
                    "recovery_steps": [step.model_dump(mode="json") for step in result.recovery_steps],
                    "evidence_list": result.evidence_list,
                    "citation_validation": result.citation_validation,
                    "trace_id": result.trace_id,
                }
            )
    finally:
        await service.aclose()

    write_jsonl(output_path, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KLERK evaluation predictions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.dataset, args.output))
