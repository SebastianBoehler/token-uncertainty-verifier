from token_uncertainty.chat_mode import (
    append_assistant,
    append_display_turn,
    clear_chat_outputs,
    messages_with_user,
)


def test_messages_with_user_preserves_history_and_adds_trimmed_user():
    history = [{"role": "assistant", "content": "Ask me something factual."}]

    messages = messages_with_user(history, "  When did Apollo 11 land?  ")

    assert messages[0] == history[0]
    assert messages[-1] == {"role": "user", "content": "When did Apollo 11 land?"}


def test_append_assistant_returns_new_history():
    messages = [{"role": "user", "content": "When?"}]

    next_history = append_assistant(messages, "Apollo 11 landed in 1969.")

    assert next_history == [
        {"role": "user", "content": "When?"},
        {"role": "assistant", "content": "Apollo 11 landed in 1969."},
    ]
    assert messages == [{"role": "user", "content": "When?"}]


def test_append_display_turn_escapes_user_text_but_keeps_overlay_html():
    display = append_display_turn([], "<b>When?</b>", "<div class='overlay'>Answer</div>")

    assert display == [
        {"role": "user", "content": "&lt;b&gt;When?&lt;/b&gt;"},
        {"role": "assistant", "content": "<div class='overlay'>Answer</div>"},
    ]


def test_append_display_turn_can_render_user_overlay_html():
    display = append_display_turn([], "<b>When?</b>", "Answer", user_html="<mark>When?</mark>")

    assert display[0]["content"] == "<mark>When?</mark>"
    assert display[1]["content"] == "Answer"


def test_clear_chat_resets_outputs():
    assert clear_chat_outputs() == ("", [], [], [], "", "", "", "", [], [])
