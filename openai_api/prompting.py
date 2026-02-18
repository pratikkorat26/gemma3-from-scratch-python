from typing import List

from .schemas import ChatMessage


def messages_to_gemma_prompt(messages: List[ChatMessage]) -> str:
    role_map = {
        "system": "user",
        "user": "user",
        "assistant": "model",
    }
    parts = []
    for message in messages:
        mapped = role_map[message.role]
        parts.append(f"<start_of_turn>{mapped}\n{message.content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)
