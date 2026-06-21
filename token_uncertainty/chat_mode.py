from __future__ import annotations

from html import escape


ChatMessage = dict[str, str]


def messages_with_user(history: list[ChatMessage] | None, message: str) -> list[ChatMessage]:
    return [*(history or []), {"role": "user", "content": message.strip()}]


def append_assistant(messages: list[ChatMessage], response: str) -> list[ChatMessage]:
    return [*messages, {"role": "assistant", "content": response}]


def append_display_turn(
    display_history: list[ChatMessage] | None,
    user_message: str,
    assistant_html: str,
    user_html: str | None = None,
) -> list[ChatMessage]:
    return [
        *(display_history or []),
        {"role": "user", "content": user_html if user_html is not None else escape(user_message.strip())},
        {"role": "assistant", "content": assistant_html},
    ]


def clear_chat_outputs():
    return "", [], [], [], "", "", "", "", [], []
