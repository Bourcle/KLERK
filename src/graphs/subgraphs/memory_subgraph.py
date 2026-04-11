from langgraph.graph import END, START, StateGraph

from graphs.state import MemoryState


def build_memory_subgraph(memory_repo, settings):
    async def load_memories(state: MemoryState):
        route = state.get("route") or {}
        memories = memory_repo.search(
            user_id=state["user_id"],
            query=state["question"],
            domain=str(route.get("source_type", "law")),
            top_k=settings.memory_top_k,
        )
        return {"memories": [memory.model_dump(mode="json") for memory in memories]}

    builder = StateGraph(MemoryState)
    builder.add_node("load_memories", load_memories)
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", END)
    return builder.compile()
