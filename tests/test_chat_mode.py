from token_uncertainty.chat_mode import append_assistant, clear_chat_outputs, messages_with_user


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


def test_clear_chat_resets_outputs():
    assert clear_chat_outputs() == ("", [], [], "", "", "", [], [])
