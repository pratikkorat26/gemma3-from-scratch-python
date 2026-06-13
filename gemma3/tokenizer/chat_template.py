def apply_chat_template(user_text: str) -> str:
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
