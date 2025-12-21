import sys
from src.server import OllamaServer
from src.model import OllamaModel
from src.chat_session import ChatSession
from src.utilities import generate_human_id
from src.llama_art import LLAMA, GOODBYE_LLAMA

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

    # Return the response and the session name
    return {"response": response, "session_name": session.session_name}


def graceful_exit(session_name=None):
    """Handle graceful exit with optional conversation summarization."""
    print(GOODBYE_LLAMA)
    if session_name:
        try:
            session_info = ChatSession.get_session_info(session_name)
            model_name = session_info["model"]
            server = OllamaServer()
            model = OllamaModel(model_name, server)
            session = ChatSession(model, session_name)
            session.summarize_conversation()
        except Exception as e:
            print(f"Warning: Could not generate summary: {e}")
        print(f"Conversation saved, byeee")
    
    sys.exit(0)

def get_input(prompt_text, session_name=None):
    """Get user input with automatic exit handling for 'quit', 'exit', and Ctrl+C."""
    try:
        user_input = input(prompt_text).strip()
        if user_input.lower() in ['quit', 'exit']:
            graceful_exit(session_name)
        return user_input
    except (KeyboardInterrupt, EOFError):
        graceful_exit(session_name)

def main():
    """
    Interactive CLI for chatting with Ollama models.

    Allows users to start new chat sessions or resume existing ones.
    """
    import os
    import sys

    print(LLAMA)
    print("-" * 50)

    # Main menu loop
    while True:
        # Ask user if they want a new chat or to resume
        print("\nWhat would you like to do?")
        print("1. Start a new chat")
        print("2. Resume an existing chat")
        print("3. Delete a conversation")

        choice = get_input("\nEnter your choice (1-3): ")

        # Handle delete conversation option
        if choice == "3":
            # Delete a conversation
            print("\n--- Delete a Conversation ---")

            conversations_dir = ".conversations"
            if not os.path.exists(conversations_dir):
                print(f"Error: No conversations directory found at {conversations_dir}")
                continue

            # Get list of session files with their summaries
            session_files = [f.replace('.json', '') for f in os.listdir(conversations_dir)
                            if f.endswith('.json')]

            if not session_files:
                print("No existing conversations found.")
                continue

            # Load session info including summaries
            session_info_list = []
            for session_file in session_files:
                try:
                    info = ChatSession.get_session_info(session_file)
                    if info:
                        summary = info.get('conversation_summary')
                        # Use summary if it exists and is not empty, otherwise use session name
                        if summary:
                            display_text = summary
                        else:
                            display_text = session_file
                        session_info_list.append({
                            'name': session_file,
                            'summary': display_text
                        })
                    else:
                        # Fallback if get_session_info returns None
                        session_info_list.append({
                            'name': session_file,
                            'summary': session_file
                        })
                except Exception as e:
                    # Fallback on error
                    print(f"Debug: Error loading {session_file}: {e}")
                    session_info_list.append({
                        'name': session_file,
                        'summary': session_file
                    })

            # Display sessions with summaries
            print("\nAvailable conversations:")
            for idx, session_info in enumerate(session_info_list, 1):
                print(f"{idx}. {session_info['summary']}")

            # Let user select which conversation to delete
            while True:
                selection = get_input(f"\nSelect a conversation to delete (1-{len(session_info_list)}) or 'back' to return: ")

                # Check if user wants to go back
                if selection.lower() == 'back':
                    print("\nReturning to main menu...")
                    break

                try:
                    selected_idx = int(selection) - 1
                    if 0 <= selected_idx < len(session_info_list):
                        session_to_delete = session_info_list[selected_idx]['name']
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(session_info_list)}")
                except ValueError:
                    print("Invalid input. Please enter a number or 'back'.")

            # If user chose 'back', skip confirmation and return to main menu
            if selection.lower() == 'back':
                continue

            # Ask for confirmation
            print(f"\nYou are about to delete: {session_info_list[selected_idx]['summary']}")
            confirmation = get_input("Are you sure you want to delete this conversation? (yes/no): ")

            if confirmation.lower() in ['yes', 'y']:
                if ChatSession.delete_session(session_to_delete):
                    print(f"\nConversation '{session_to_delete}' has been deleted.")
                else:
                    print(f"\nError: Could not delete conversation '{session_to_delete}'.")
            else:
                print("\nDeletion cancelled.")

            # Return to main menu
            continue

        # Break out of menu loop for option 1
        elif choice == "1":
            break
        # Handle option 2 inside the loop
        elif choice == "2":
            # Resume existing chat - handle inside loop, break when session is selected
            pass  # Will be handled below
        else:
            print("Invalid choice. Please try again.")
            continue

        # Option 2 handling (inside the main menu loop)
        if choice == "2":
            # Resume existing chat
            print("\n--- Resume Existing Chat ---")

            conversations_dir = ".conversations"
            if not os.path.exists(conversations_dir):
                print(f"Error: No conversations directory found at {conversations_dir}")
                continue

            # Get list of session files with their summaries
            session_files = [f.replace('.json', '') for f in os.listdir(conversations_dir)
                            if f.endswith('.json')]

            if not session_files:
                print("No existing conversations found.")
                continue

            # Load session info including summaries
            session_info_list = []
            for session_file in session_files:
                try:
                    info = ChatSession.get_session_info(session_file)
                    if info:
                        summary = info.get('conversation_summary')
                        # Use summary if it exists and is not empty, otherwise use session name
                        if summary:
                            display_text = summary
                        else:
                            display_text = session_file
                        session_info_list.append({
                            'name': session_file,
                            'summary': display_text
                        })
                    else:
                        # Fallback if get_session_info returns None
                        session_info_list.append({
                            'name': session_file,
                            'summary': session_file
                        })
                except Exception as e:
                    # Fallback on error
                    print(f"Debug: Error loading {session_file}: {e}")
                    session_info_list.append({
                        'name': session_file,
                        'summary': session_file
                    })

            # Display sessions with summaries
            print("\nAvailable conversations:")
            for idx, session_info in enumerate(session_info_list, 1):
                print(f"{idx}. {session_info['summary']}")

            # Let user select
            session_selected = False
            while True:
                selection = get_input(f"\nSelect a conversation (1-{len(session_info_list)}) or 'back' to return: ")

                # Check if user wants to go back
                if selection.lower() == 'back':
                    print("\nReturning to main menu...")
                    break

                try:
                    selected_idx = int(selection) - 1
                    if 0 <= selected_idx < len(session_info_list):
                        session_name = session_info_list[selected_idx]['name']
                        session_selected = True
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(session_info_list)}")
                except ValueError:
                    print("Invalid input. Please enter a number or 'back'.")

            # If user chose 'back', return to main menu
            if not session_selected:
                continue

            print(f"\nResuming conversation: {session_name}")

            # Print conversation history to help user feel back in context
            try:
                session_information = ChatSession.get_session_info(session_name)
                session_model = session_information["model"]

                # Instantiate objects to load the session
                server = OllamaServer()
                model = OllamaModel(session_model, server)
                temp_session = ChatSession(model, session_name)

                # Print the conversation history
                temp_session.print_conversation()
            except Exception as e:
                print(f"Warning: Could not load conversation history: {e}")

            # Break out of main menu loop to start chatting
            break

    # Handle option 1 (outside the loop, after break)
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
                selection = get_input(f"\nSelect a model (1-{len(installed_models)}): ")
                try:
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
            model_name = get_input("Enter model name (e.g., 'smollm2:360m', 'llama2'): ")
            if not model_name:
                model_name = "smollm2:360m"
                print(f"Using default model: {model_name}")

        system_prompt = get_input("Enter system prompt (press Enter for default): ")
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
            print(f"Using default system prompt: {system_prompt}")

        session_name = None

    # Main chat loop
    print("\n" + "=" * 50)
    print("Chat started! Type 'exit' or 'quit' to end the conversation.")
    print("=" * 50 + "\n")

    while True:
        # Get user input (handles exit/quit/Ctrl+C automatically)
        user_input = get_input("👤 You: ", session_name)

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
                # Get model name for resumed sessions
                if 'model_name' not in locals():
                    session_info = ChatSession.get_session_info(session_name)
                    model_name = session_info.get('model', 'unknown')

            print(f"\n🤖 Assistant: {response}")
            print(f"\n{'─' * 40}")
            print(f"  model: {model_name}")
            print(f"{'─' * 40}\n")

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
