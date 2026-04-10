from typing import TypedDict


class ReviewState(TypedDict):
    """Shared state passed through the LangGraph review workflow."""

    code: str
    language: str
    peer_review: str
    manager_review: str
