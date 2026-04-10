"""Unit tests for the CodeOptimizer review workflow.

Ollama is mocked so that these tests run offline without a local model.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents import create_peer_reviewer, create_manager_reviewer
from src.state import ReviewState
from src.workflow import create_review_workflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CODE = """\
def add(a, b):
    return a + b
"""


def _mock_llm_response(content: str) -> MagicMock:
    """Return a mock ChatOllama that always replies with *content*."""
    mock_response = MagicMock()
    mock_response.content = content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestPeerReviewer:
    @patch("src.agents.ChatOllama")
    def test_returns_peer_review_key(self, mock_chat_ollama):
        mock_chat_ollama.return_value = _mock_llm_response("Looks good!")

        reviewer = create_peer_reviewer(model="llama3.2")
        state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "Python",
            "peer_review": "",
            "manager_review": "",
        }
        result = reviewer(state)

        assert "peer_review" in result
        assert result["peer_review"] == "Looks good!"

    @patch("src.agents.ChatOllama")
    def test_prompt_contains_code(self, mock_chat_ollama):
        mock_llm = _mock_llm_response("ok")
        mock_chat_ollama.return_value = mock_llm

        reviewer = create_peer_reviewer()
        state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "",
            "peer_review": "",
            "manager_review": "",
        }
        reviewer(state)

        called_prompt = mock_llm.invoke.call_args[0][0]
        assert SAMPLE_CODE in called_prompt

    @patch("src.agents.ChatOllama")
    def test_prompt_includes_language_hint(self, mock_chat_ollama):
        mock_llm = _mock_llm_response("ok")
        mock_chat_ollama.return_value = mock_llm

        reviewer = create_peer_reviewer()
        state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "Python",
            "peer_review": "",
            "manager_review": "",
        }
        reviewer(state)

        called_prompt = mock_llm.invoke.call_args[0][0]
        assert "Python" in called_prompt


class TestManagerReviewer:
    @patch("src.agents.ChatOllama")
    def test_returns_manager_review_key(self, mock_chat_ollama):
        mock_chat_ollama.return_value = _mock_llm_response("Ship it.")

        reviewer = create_manager_reviewer(model="llama3.2")
        state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "Python",
            "peer_review": "Minor nits only.",
            "manager_review": "",
        }
        result = reviewer(state)

        assert "manager_review" in result
        assert result["manager_review"] == "Ship it."

    @patch("src.agents.ChatOllama")
    def test_prompt_includes_peer_review(self, mock_chat_ollama):
        mock_llm = _mock_llm_response("ok")
        mock_chat_ollama.return_value = mock_llm

        reviewer = create_manager_reviewer()
        peer_review_text = "Watch out for integer overflow."
        state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "",
            "peer_review": peer_review_text,
            "manager_review": "",
        }
        reviewer(state)

        called_prompt = mock_llm.invoke.call_args[0][0]
        assert peer_review_text in called_prompt


# ---------------------------------------------------------------------------
# Workflow / graph tests
# ---------------------------------------------------------------------------


class TestReviewWorkflow:
    @patch("src.agents.ChatOllama")
    def test_workflow_compiles_and_runs(self, mock_chat_ollama):
        """The compiled graph should traverse both nodes and populate state."""
        call_count = 0
        responses = ["Peer feedback here.", "Manager feedback here."]

        def side_effect(*args, **kwargs):
            nonlocal call_count
            mock_llm = _mock_llm_response(responses[call_count % len(responses)])
            call_count += 1
            return mock_llm

        mock_chat_ollama.side_effect = side_effect

        workflow = create_review_workflow(model="llama3.2")
        initial_state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "Python",
            "peer_review": "",
            "manager_review": "",
        }
        result = workflow.invoke(initial_state)

        assert result["peer_review"] == "Peer feedback here."
        assert result["manager_review"] == "Manager feedback here."

    @patch("src.agents.ChatOllama")
    def test_workflow_preserves_code_in_state(self, mock_chat_ollama):
        mock_chat_ollama.side_effect = lambda *a, **kw: _mock_llm_response("ok")

        workflow = create_review_workflow()
        initial_state: ReviewState = {
            "code": SAMPLE_CODE,
            "language": "",
            "peer_review": "",
            "manager_review": "",
        }
        result = workflow.invoke(initial_state)

        assert result["code"] == SAMPLE_CODE

    @patch("src.agents.ChatOllama")
    def test_workflow_accepts_custom_model_and_url(self, mock_chat_ollama):
        mock_chat_ollama.side_effect = lambda *a, **kw: _mock_llm_response("ok")

        workflow = create_review_workflow(
            model="codellama",
            base_url="http://localhost:11434",
        )
        assert workflow is not None
