from src.server import OllamaServer
from src.model import OllamaModel
from src.chat_session import ChatSession
from src.utilities import generate_human_id

def start_chat_session(prompt: str, model_name: str, system_prompt: str = None) -> dict[str, str]:
    """
    Start a new conversation with an Ollama model.

    Creates a new chat session with a generated session ID, sets the system prompt,
    and sends the initial prompt to the model.

    Args:
        prompt: The initial user prompt to send to the model.
        model_name: The name of the Ollama model to use (e.g., 'smollm2:360m', 'llama2').
        system_prompt: Instructions that guide the AI's behavior and response style.

    Returns:
        A dictionary containing:
            - 'response': The model's response text to the initial prompt
            - 'session_name': The unique identifier for this conversation session

    Example:
        >>> result = start_conversation("Hello", "smollm2:360m", "You are helpful")
        >>> print(result['response'])
        >>> print(result['session_name'])
    """

    # Instantiate OllamaServer and OllamaModel
    server = OllamaServer()
    model = OllamaModel(model_name, server)
    
    # Instantiate a new ChatSession
    session = ChatSession(model, generate_human_id())

    # Set up the system prompt (ie. instructions for the AI response behaviour)
    session.set_system_prompt(system_prompt)

    # Send a new message
    response = session.send_message(prompt)

    # Summarize the conversation
    session.summarize_conversation()
    
    # Return the response and the session name
    return {"response": response, "session_name": session.session_name}
    
def continue_chat_session(prompt: str, session_name: str) -> dict[str, str]:
    """
    Continue an existing conversation with an Ollama model.

    Retrieves a previous chat session by its name, restores the model context,
    and sends a new prompt while maintaining conversation history.

    Args:
        prompt: The user's message to send in the ongoing conversation.
        session_name: The unique identifier of the session to continue.

    Returns:
        A dictionary containing:
            - 'response': The model's response text to the prompt
            - 'session_name': The unique identifier for this conversation session

    Raises:
        ValueError: If the session_name does not correspond to an existing session.

    Example:
        >>> response = continue_conversation("Tell me more", "session_abc123")
        >>> print(response)
    """

    # Retrieve session information
    session_information = ChatSession.get_session_info(session_name)
    session_model = session_information["model"]

    # Instantiate objects
    server = OllamaServer()
    model = OllamaModel(session_model, server)

    # Restore the session
    session = ChatSession(model, session_name)

    # Send new message
    response = session.send_message(prompt)

    # Summarize the conversation
    session.summarize_conversation()

    # Return the response and the session name
    return {"response": response, "session_name": session.session_name}


def main():
    """
    Interactive CLI for chatting with Ollama models.

    Allows users to start new chat sessions or resume existing ones.
    """
    import os
    import sys

    print("Welcome to Ollama Chat!")
    print("-" * 50)

    # Ask user if they want a new chat or to resume
    print("\nWhat would you like to do?")
    print("1. Start a new chat")
    print("2. Resume an existing chat")

    choice = input("\nEnter your choice (1-2): ").strip()

    if choice == "1":
        # Start a new chat
        print("\n--- Starting New Chat ---")

        # Get list of installed models
        try:
            server = OllamaServer()
            installed_models = sorted(server.get_installed_models())

            if not installed_models:
                print("No models found. Please install a model first.")
                sys.exit(1)

            # Display models with numbers
            print("\nAvailable models:")
            for idx, model in enumerate(installed_models, 1):
                print(f"{idx}. {model}")

            # Let user select
            while True:
                try:
                    selection = input(f"\nSelect a model (1-{len(installed_models)}): ").strip()
                    selected_idx = int(selection) - 1
                    if 0 <= selected_idx < len(installed_models):
                        model_name = installed_models[selected_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(installed_models)}")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            print(f"\nUsing model: {model_name}")

        except Exception as e:
            print(f"Error fetching models: {e}")
            print("Falling back to manual entry.")
            model_name = input("Enter model name (e.g., 'smollm2:360m', 'llama2'): ").strip()
            if not model_name:
                model_name = "smollm2:360m"
                print(f"Using default model: {model_name}")

        system_prompt = input("Enter system prompt (press Enter for default): ").strip()
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
            print(f"Using default system prompt: {system_prompt}")

        session_name = None

    elif choice == "2":
        # Resume existing chat
        print("\n--- Resume Existing Chat ---")

        conversations_dir = ".conversations"
        if not os.path.exists(conversations_dir):
            print(f"Error: No conversations directory found at {conversations_dir}")
            sys.exit(1)

        # Get list of session files
        session_files = [f.replace('.json', '') for f in os.listdir(conversations_dir)
                        if f.endswith('.json')]

        if not session_files:
            print("No existing conversations found.")
            sys.exit(1)

        # Display sessions with numbers
        print("\nAvailable conversations:")
        for idx, session in enumerate(session_files, 1):
            print(f"{idx}. {session}")

        # Let user select
        while True:
            try:
                selection = input(f"\nSelect a conversation (1-{len(session_files)}): ").strip()
                selected_idx = int(selection) - 1
                if 0 <= selected_idx < len(session_files):
                    session_name = session_files[selected_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(session_files)}")
            except (ValueError, KeyboardInterrupt):
                print("\nInvalid input. Please enter a number.")

        print(f"\nResuming conversation: {session_name}")

    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Main chat loop
    print("\n" + "=" * 50)
    print("Chat started! Type 'exit' or 'quit' to end the conversation.")
    print("=" * 50 + "\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nEnding chat session. Goodbye!")
                if session_name:
                    print(f"Your conversation has been saved as: {session_name}")
                break

            if not user_input:
                continue

            # Send message and get response
            try:
                if session_name is None:
                    # First message of a new chat
                    result = start_chat_session(user_input, model_name, system_prompt)
                    session_name = result['session_name']
                    response = result['response']
                    print(f"\n[Session created: {session_name}]")
                else:
                    # Continuing chat
                    result = continue_chat_session(user_input, session_name)
                    response = result['response']

                print(f"\nAssistant: {response}\n")

            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")

    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye!")
        if session_name:
            print(f"Your conversation has been saved as: {session_name}")


if __name__ == "__main__":
    main()
