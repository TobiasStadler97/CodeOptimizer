"""Agent node factories for the code review workflow.

Each factory returns a callable that:
  - Accepts the current :class:`~src.state.ReviewState`
  - Returns a *partial* state dict whose keys will be merged into the
    running graph state by LangGraph.
"""

from langchain_ollama import ChatOllama

from src.state import ReviewState

_PEER_SYSTEM = (
    "You are an experienced software engineer performing a peer code review. "
    "Be thorough, constructive and specific."
)

_MANAGER_SYSTEM = (
    "You are a senior engineering manager reviewing code and an accompanying "
    "peer-review report. Provide high-level, strategic feedback."
)


def create_peer_reviewer(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
):
    """Return a LangGraph node that performs a peer code review.

    Args:
        model: Ollama model tag (e.g. ``"llama3.2"``, ``"codellama"``).
        base_url: Base URL of the running Ollama instance.

    Returns:
        A callable suitable for use as a LangGraph node.
    """
    llm = ChatOllama(model=model, base_url=base_url)

    def peer_reviewer(state: ReviewState) -> dict:
        lang_hint = f" {state['language']}" if state.get("language") else ""
        prompt = (
            f"{_PEER_SYSTEM}\n\n"
            f"Please review the following{lang_hint} code and provide detailed "
            "feedback on:\n"
            "1. Code quality and readability\n"
            "2. Potential bugs or edge cases\n"
            "3. Performance considerations\n"
            "4. Best practices and conventions\n"
            "5. Concrete improvement suggestions\n\n"
            f"Code:\n```\n{state['code']}\n```"
        )
        response = llm.invoke(prompt)
        return {"peer_review": response.content}

    return peer_reviewer


def create_manager_reviewer(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
):
    """Return a LangGraph node that performs a manager code review.

    The manager sees both the original code *and* the peer review produced
    by the previous node in the graph.

    Args:
        model: Ollama model tag.
        base_url: Base URL of the running Ollama instance.

    Returns:
        A callable suitable for use as a LangGraph node.
    """
    llm = ChatOllama(model=model, base_url=base_url)

    def manager_reviewer(state: ReviewState) -> dict:
        lang_hint = f" {state['language']}" if state.get("language") else ""
        prompt = (
            f"{_MANAGER_SYSTEM}\n\n"
            f"The following{lang_hint} code has already received a peer review. "
            "Please provide a high-level manager review focusing on:\n"
            "1. Overall architecture and design decisions\n"
            "2. Technical debt and maintainability\n"
            "3. Alignment with team/business standards\n"
            "4. Risk assessment\n"
            "5. Priority recommendations for the author\n\n"
            f"Code:\n```\n{state['code']}\n```\n\n"
            f"Peer Review:\n{state.get('peer_review', 'Not available')}"
        )
        response = llm.invoke(prompt)
        return {"manager_review": response.content}

    return manager_reviewer
