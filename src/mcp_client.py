from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from utils.config import Settings
from utils.exceptions import ConfigError, MCPError
from parsers import (
    chunk_text,
    extract_titles,
    parse_constitutional_ids,
    parse_law_ids,
    parse_precedent_ids,
)
from data_structure.schemas import MCPFetchResult, RouteDecision


def tool_output_to_text(result: Any) -> str:
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)
    if isinstance(result, dict) and "content" in result:
        content = result["content"]
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)
    return str(result)


class KoreanLawMCPGateway:
    SERVER_NAME = "korean_law"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = MultiServerMCPClient(
            {
                self.SERVER_NAME: {
                    "transport": "stdio",
                    "command": self.settings.mcp_server_command,
                    "args": [self.settings.mcp_server_entrypoint],
                    "env": {"LAW_API_OC": self.settings.law_api_oc or ""},
                }
            }
        )

    def is_enabled(self) -> bool:
        return bool(self.settings.law_api_oc)

    async def _call_tool(self, tools: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> str:
        tool = tools.get(tool_name)
        if tool is None:
            raise MCPError(f"Tool not found: {tool_name}")
        try:
            result = await tool.ainvoke(arguments)
            text = tool_output_to_text(result)
            if "isError" in text or ("error" in text.lower() and "id" not in text.lower()):
                raise MCPError(text)
            return text
        except MCPError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPError(f"{tool_name} failed: {exc}") from exc

    async def search_and_fetch(self, *, route: RouteDecision, query: str, fetch_top_n: int = 3) -> list[MCPFetchResult]:
        if not self.is_enabled():
            raise ConfigError("LAW_API_OC is not set. MCP fallback is disabled.")

        if route.source_type == "law":
            search_name = "search_korean_law"
            detail_name = "get_law_detail"
            id_parser = parse_law_ids
        elif route.source_type == "precedent":
            search_name = "search_precedent"
            detail_name = "get_precedent_detail"
            id_parser = parse_precedent_ids
        else:
            search_name = "search_constitutional"
            detail_name = "get_constitutional_detail"
            id_parser = parse_constitutional_ids

        async with self.client.session(self.SERVER_NAME) as session:
            tools = {tool.name: tool for tool in await load_mcp_tools(session)}

            search_text = await self._call_tool(
                tools,
                search_name,
                {"query": query, "page": 1, "display": fetch_top_n},
            )
            raw_ids = id_parser(search_text)[:fetch_top_n]
            titles = extract_titles(search_text)

            if not raw_ids:
                if search_text.strip():
                    return [
                        MCPFetchResult(
                            source_type=route.source_type,
                            title=titles[0] if titles else query,
                            raw_id="search_only",
                            content=search_text,
                            metadata={"search_only": True, "search_tool": search_name},
                        )
                    ]
                return []

            collected: list[MCPFetchResult] = []
            for idx, raw_id in enumerate(raw_ids):
                detail_text = await self._call_tool(tools, detail_name, self._detail_args(detail_name, raw_id))
                title = titles[idx] if idx < len(titles) else query
                for chunk_idx, chunk in enumerate(chunk_text(detail_text)):
                    collected.append(
                        MCPFetchResult(
                            source_type=route.source_type,
                            title=title,
                            raw_id=f"{raw_id}:{chunk_idx}",
                            content=chunk,
                            metadata={
                                "origin_id": raw_id,
                                "search_tool": search_name,
                                "detail_tool": detail_name,
                                "chunk_index": chunk_idx,
                            },
                        )
                    )
            return collected

    @staticmethod
    def _detail_args(tool_name: str, raw_id: str) -> dict[str, str]:
        if tool_name == "get_law_detail":
            return {"lawId": raw_id}
        if tool_name == "get_precedent_detail":
            return {"precId": raw_id}
        if tool_name == "get_constitutional_detail":
            return {"decisionId": raw_id}
        raise ValueError(f"Unsupported detail tool: {tool_name}")
