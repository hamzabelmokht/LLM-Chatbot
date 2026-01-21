SYSTEM_PROMPT = "You are a helpful AI tutor."


def build_prompt(conversation):
    prompt = f"System: {SYSTEM_PROMPT}\n\n"

    for message in conversation:
        role = message["role"].capitalize()
        content = message["content"]
        prompt += f"{role}: {content}\n"

    prompt += "Assistant:"
    return prompt
