import google.generativeai as genai
import re
import os
import json

# --- CONFIGURATION ---
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
API_KEY = config_data["GEMINI_API_KEY"]


# Configure the library
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")


def preprocess_input(user_text):

    # 1. Lowercasing
    text = user_text.lower()

    # 2. Punctuation removal (using Regex)
    text = re.sub(r"[^\w\s]", "", text)

    # 3. Tokenization (Creating a list of tokens)
    tokens = text.split()

    # For the purpose of the API, we rejoin them,
    # but we return tokens to prove preprocessing happened.
    processed_text = " ".join(tokens)

    return processed_text, tokens


def main():
    print("--- LLM Q&A CLI Application ---")
    print("Type 'exit' or 'quit' to stop the program.\n")

    while True:
        user_question = input("Enter your question: ")

        if user_question.lower() in ["exit", "quit"]:
            print("Exiting application. Goodbye!")
            break

        if not user_question.strip():
            print("Please enter a valid question.")
            continue

        try:
            print("Processing...")

            # Preprocessing Step
            clean_query, tokens = preprocess_input(user_question)

            # Send to LLM API
            response = model.generate_content(clean_query)

            # Display Results
            print(f"\n[Processed Query]: {clean_query}")
            print(f"[Tokens Detected]: {tokens}")
            print(f"\n[LLM Answer]:\n{response.text}")
            print("-" * 50)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
