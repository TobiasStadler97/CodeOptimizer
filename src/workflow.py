"""LangGraph workflow for the two-stage code review pipeline.

Graph topology::

    START --> peer_reviewer --> manager_reviewer --> END
"""

from langgraph.graph import END, START, StateGraph

from src.agents import create_peer_reviewer, create_manager_reviewer
from src.state import ReviewState


def create_review_workflow(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
):
    """Build and compile the code review :class:`~langgraph.graph.StateGraph`.

    The graph runs two nodes in sequence:

    1. **peer_reviewer** – reviews code quality, style and correctness.
    2. **manager_reviewer** – reviews architecture, risk and priorities,
       taking the peer review into account.

    Args:
        model: Ollama model tag to use for both agents
               (e.g. ``"llama3.2"``, ``"codellama"``).
        base_url: Base URL of the running Ollama instance.
                  Defaults to ``"http://localhost:11434"``.

    Returns:
        A compiled LangGraph runnable that accepts a :class:`~src.state.ReviewState`
        and returns the fully populated state after both reviews.
    """
    peer_reviewer = create_peer_reviewer(model=model, base_url=base_url)
    manager_reviewer = create_manager_reviewer(model=model, base_url=base_url)

    builder = StateGraph(ReviewState)

    builder.add_node("peer_reviewer", peer_reviewer)
    builder.add_node("manager_reviewer", manager_reviewer)

    builder.add_edge(START, "peer_reviewer")
    builder.add_edge("peer_reviewer", "manager_reviewer")
    builder.add_edge("manager_reviewer", END)

    return builder.compile()
