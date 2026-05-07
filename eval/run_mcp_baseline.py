import argparse
import asyncio

from dotenv import load_dotenv

from src.answering import generate_answer
from src.data_structure.schemas import RetrievedChunk
from eval.dataset import load_jsonl, write_jsonl
from src.llm_model.llm import LLMFactory
from src.mcp_client import KoreanLawMCPGateway
from src.router import heuristic_route
from src.utils.citations import attach_evidence_ids, validate_citations
from src.utils.config import get_settings


async def run(dataset_path: str, output_path: str) -> None:
    """Run the MCP-only baseline evaluation and write prediction outputs.

    Args:
        dataset_path: Path to the input evaluation JSONL dataset.
        output_path: Path where generated baseline outputs should be written.

    Returns:
        None: This function completes after writing all baseline prediction rows.

    Raises:
        RuntimeError: If LAW_API_OC is not configured for MCP access.
    """

    load_dotenv(override=True)
    settings = get_settings()
    if not settings.law_api_oc:
        raise RuntimeError("LAW_API_OC is not set. MCP-only baseline cannot run.")

    factory = LLMFactory(settings)
    factory.validate_settings()
    model = factory.create_chat_model()
    gateway = KoreanLawMCPGateway(settings)
    rows = load_jsonl(dataset_path)
    outputs = []

    for item in rows:
        route = heuristic_route(item["question"], settings)
        mcp_results = await gateway.search_and_fetch(
            route=route,
            query=item["question"],
            fetch_top_n=settings.mcp_fetch_top_n,
        )
        docs = [
            RetrievedChunk(
                content=result.content,
                similarity=0.0,
                source="mcp_only",
                title=result.title,
                source_id=result.raw_id,
                collection=route.collection,
                metadata={**result.metadata, "source_type": result.source_type},
            )
            for result in mcp_results
        ]
        docs, evidence_list = attach_evidence_ids(docs)
        answer = await generate_answer(
            model=model,
            settings=settings,
            question=item["question"],
            route=route,
            memories=[],
            docs=docs,
            evidence_sufficient=bool(docs),
        )
        outputs.append(
            {
                "id": item["id"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "reference_sources": item.get("reference_sources", []),
                "prediction": answer,
                "route": route.model_dump(mode="json"),
                "selected_domain": route.topic,
                "selected_collection": route.collection,
                "used_mcp": True,
                "retrieval_sufficient": bool(docs),
                "sufficiency_reason": "MCP-only baseline does not run KLERK sufficiency judge",
                "recovery_steps": [],
                "evidence_list": evidence_list,
                "citation_validation": validate_citations(answer, evidence_list),
            }
        )

    write_jsonl(output_path, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP-only baseline predictions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.dataset, args.output))
