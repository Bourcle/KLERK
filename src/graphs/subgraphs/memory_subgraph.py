from langgraph.graph import END, START, StateGraph

from graphs.state import MemoryState


def build_memory_subgraph(memory_repo, settings):
    """Build and compile the memory-loading subgraph.

    Args:
        memory_repo: Repository used to search user memories by question and domain.
        settings: Runtime settings containing memory retrieval options.

    Returns:
        CompiledStateGraph: A compiled LangGraph runnable for loading relevant memories.
    """

    async def load_memories(state: MemoryState):
        """Load relevant user memories for the current question and route.

        Args:
            state: Current memory graph state containing user ID, question, and route metadata.

        Returns:
            dict: State updates containing serialized memory items.
        """

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
