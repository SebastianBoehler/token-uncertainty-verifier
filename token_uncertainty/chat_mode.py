from __future__ import annotations


ChatMessage = dict[str, str]


def messages_with_user(history: list[ChatMessage] | None, message: str) -> list[ChatMessage]:
    return [*(history or []), {"role": "user", "content": message.strip()}]


def append_assistant(messages: list[ChatMessage], response: str) -> list[ChatMessage]:
    return [*messages, {"role": "assistant", "content": response}]


def clear_chat_outputs():
    return "", [], [], "", "", "", [], []
