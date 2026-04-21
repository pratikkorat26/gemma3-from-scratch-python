import re
from typing import List

from .schemas import ChatMessage

DANGEROUS_TOKENS = re.compile(
    r"<start_of_turn>|<end_of_turn>|<eos>|<bos>|<func_call>|<receiver_bot>",
    re.IGNORECASE,
)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_CONTENT_LENGTH = 32_768


def sanitize_user_content(content: str) -> str:
    content = CONTROL_CHARS.sub("", content)
    content = DANGEROUS_TOKENS.sub("", content)
    return content


def messages_to_gemma_prompt(messages: List[ChatMessage]) -> str:
    role_map = {
        "system": "user",
        "user": "user",
        "assistant": "model",
    }
    parts = []
    for message in messages:
        mapped = role_map[message.role]
        sanitized = sanitize_user_content(message.content)
        if len(sanitized) > MAX_CONTENT_LENGTH:
            sanitized = sanitized[:MAX_CONTENT_LENGTH]
        parts.append(f"<start_of_turn>{mapped}\n{sanitized}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)
