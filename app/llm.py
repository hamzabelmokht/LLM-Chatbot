import requests
import json

# -----------------------------
# Configuration
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "phi"


def stream_llm(prompt: str):
    """
    Generator that yields tokens from the LLM as they arrive.
    """

    payload = {
        "model": MODEL_NAME,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "num_predict": 150
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        if "message" in data and "content" in data["message"]:
            yield data["message"]["content"]

        if data.get("done", False):
            break


def call_llm(prompt: str) -> str:
    """
    Convenience wrapper that returns the full response as a string.
    """
    full_response = ""
    for token in stream_llm(prompt):
        full_response += token
    return full_response
