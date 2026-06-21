from token_uncertainty import app_actions
from token_uncertainty.types import AnalysisResult, TokenScore


def test_run_chat_turn_appends_assistant_and_analyzes(monkeypatch):
    def fake_generate_chat_with_scores(**kwargs):
        assert kwargs["messages"][-1]["content"] == "When did Apollo 11 land?"
        return AnalysisResult(
            text="Apollo 11 landed in 1969.",
            tokens=[TokenScore(text="1969", uncertainty=0.95)],
            sentences=[],
        )

    monkeypatch.setattr(app_actions, "generate_chat_with_scores", fake_generate_chat_with_scores)

    message, chatbot, state, analyzed_text, token_html, *_ = app_actions.run_chat_turn(
        "When did Apollo 11 land?",
        [],
        "test-model",
        32,
        0.1,
        0.9,
        0.82,
    )

    assert message == ""
    assert chatbot == state
    assert state[-2]["role"] == "user"
    assert state[-1]["role"] == "assistant"
    assert analyzed_text == "Apollo 11 landed in 1969."
    assert "1969" in token_html


def test_clear_chat_resets_outputs():
    assert app_actions.clear_chat() == ("", [], [], "", "", "", [], [])
