from llm import call_llm
from memory import add_message, get_conversation
from prompting import build_prompt


def run_chat():
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        add_message("user", user_input)

        prompt = build_prompt(get_conversation())

        # Print label ONCE before streaming starts
        print("Bot: ", end="", flush=True)

        response = call_llm(prompt)

        # Store response for memory only
        add_message("assistant", response)
