#!/usr/bin/env python3
"""
Example demonstrating programmatic conversation loops with context management.

This script shows how to use ChatSession to maintain conversation context
across multiple prompts without terminal input.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_ollama import ChatSession, ensure_ollama_server_running


def main():
    """Run a conversation loop with predefined prompts."""
    print("Starting Ollama conversation loop example...\n")

    # Ensure Ollama server is running
    ensure_ollama_server_running()

    # Model to use (change this to your preferred installed model)
    # To see available models: ollama list
    # To install a model: ollama pull <model-name>
    model_name = "llama2"

    # Create chat session with the model
    chat = ChatSession(model_name)

    # Optional: Set system prompt for behavior customization
    chat.set_system_prompt("You are a helpful assistant that provides clear and concise answers.")

    # Define a series of prompts for the conversation
    prompts = [
        "What is machine learning?",
        "Can you give me a simple example?",
        "How would I implement that in Python?",
        "What are some common applications?",
    ]

    print("=" * 60)
    print("CONVERSATION LOOP WITH CONTEXT MANAGEMENT")
    print("=" * 60 + "\n")

    # Process each prompt while maintaining context
    for i, prompt in enumerate(prompts, 1):
        print(f"[Turn {i}] User: {prompt}")
        print("-" * 60)

        try:
            response = chat.send_message(prompt)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
            break

    # Show conversation statistics
    history = chat.get_history()
    print("=" * 60)
    print("CONVERSATION COMPLETE")
    print("=" * 60)
    print(f"Total messages: {len(history)}")
    print(f"User messages: {sum(1 for m in history if m.get('role') == 'user')}")
    print(f"Assistant messages: {sum(1 for m in history if m.get('role') == 'assistant')}\n")

    # Show how to reset conversation for a new topic
    print("Resetting conversation for new topic...")
    chat.reset()

    # Example of new conversation with different system prompt
    chat.set_system_prompt("You are a programming expert specializing in Python.")
    response = chat.send_message("What are decorators in Python?")
    print(f"New conversation - Response: {response[:100]}...")


if __name__ == "__main__":
    main()
