import argparse
from src.models import check_and_install_model
from src.inactivity_monitor import ModelInactivityMonitor
from src.sessions import ChatSession
from src.server import OllamaServer

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run Ollama models with optional server management and inactivity monitoring'
    )

    # Model selection arguments
    parser.add_argument(
        'prompt',
        help='The prompt to send to the model'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Model name to use (required)'
    )

    # Server management arguments
    parser.add_argument(
        '--skip-server-check',
        action='store_true',
        help='Skip checking if Ollama server is running'
    )
    parser.add_argument(
        '--skip-model-check',
        action='store_true',
        help='Skip checking if model is installed'
    )

    # Inactivity monitoring arguments
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        required=True,
        help='Inactivity timeout in minutes (required)'
    )
    parser.add_argument(
        '--no-inactivity-monitor',
        action='store_true',
        help='Disable inactivity monitoring'
    )

    # Conversation mode
    parser.add_argument(
        '--conversation',
        action='store_true',
        help='Run in conversation mode (maintains context between prompts)'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        help='Optional system prompt for conversation mode'
    )
    parser.add_argument(
        '--session',
        type=str,
        help='Session name for persistent conversations (saves/loads from .conversations/)'
    )
    parser.add_argument(
        '--new-session',
        action='store_true',
        help='Start a new session (clears existing session with same name)'
    )

    # Session management commands
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List all saved conversation sessions'
    )
    parser.add_argument(
        '--show-session',
        type=str,
        help='Show details of a specific session'
    )
    parser.add_argument(
        '--clear-session',
        type=str,
        help='Delete a specific session'
    )

    args = parser.parse_args()

    # Handle session management commands first
    if args.list_sessions:
        sessions = ChatSession.list_sessions()
        if not sessions:
            print("No saved sessions found.")
        else:
            print(f"\nFound {len(sessions)} session(s):\n")
            print(f"{'Name':<20} {'Model':<15} {'Messages':<10} {'Last Updated':<20}")
            print("-" * 70)
            for session in sessions:
                print(f"{session['name']:<20} {session['model']:<15} {session['message_count']:<10} {session['last_updated']:<20}")
        return

    if args.show_session:
        info = ChatSession.get_session_info(args.show_session)
        if not info:
            print(f"Session '{args.show_session}' not found.")
            return

        print(f"\nSession: {info['name']}")
        print(f"Model: {info['model']}")
        print(f"Created: {info['created_at']}")
        print(f"Last Updated: {info['last_updated']}")
        print(f"Messages: {info['message_count']}")
        if info['system_prompt']:
            print(f"System Prompt: {info['system_prompt']}")
        print("\nConversation History:")
        print("=" * 60)
        for msg in info['messages']:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            if role == 'System':
                print(f"\n[{role}]: {content}")
            else:
                print(f"\n[{role}]: {content}")
            print("-" * 60)
        return

    if args.clear_session:
        if ChatSession.delete_session(args.clear_session):
            print(f"Session '{args.clear_session}' deleted.")
        else:
            print(f"Session '{args.clear_session}' not found.")
        return

    # Ensure Ollama server is running
    if not args.skip_server_check:
        server = OllamaServer()
        server.test_running()

    # Use provided model
    selected_model = args.model

    # Check if model is installed
    if not args.skip_model_check:
        check_and_install_model(selected_model)

    # Start inactivity monitor if not disabled
    monitor = None
    if not args.no_inactivity_monitor and args.timeout > 0:
        monitor = ModelInactivityMonitor(selected_model, args.timeout)
        monitor.start_monitoring()

    try:
        if args.conversation:
            # Conversation mode: maintains context between prompts
            if not args.prompts and not args.session:
                raise ValueError("Conversation mode requires either --prompts argument or --session with a prompt")

            # Handle session creation/loading
            session_name = args.session
            if session_name and args.new_session:
                # Clear existing session if --new-session flag is set
                if ChatSession.session_exists(session_name):
                    ChatSession.delete_session(session_name)
                    print(f"Cleared existing session '{session_name}'")

            # Create chat session with optional persistence
            chat = ChatSession(
                model_name=selected_model,
                session_name=session_name
            )

            # Check if we're continuing an existing session
            is_continuing = session_name and ChatSession.session_exists(session_name) and not args.new_session

            if is_continuing:
                history = chat.get_history()
                user_messages = [m for m in history if m.get('role') == 'user']
                print(f"Continuing session '{session_name}' ({len(user_messages)} previous exchanges)")
            else:
                # Set system prompt if provided (only for new sessions)
                if args.system_prompt:
                    chat.set_system_prompt(args.system_prompt)

            # If we have prompts to process, do so
            if args.prompts:
                print(f"\nProcessing {len(args.prompts)} prompt(s)...\n")
                print("=" * 60)

                for i, user_prompt in enumerate(args.prompts, 1):
                    print(f"\n[Turn {i}] User: {user_prompt}")
                    print("-" * 60)

                    if user_prompt.strip():
                        if monitor:
                            monitor.record_interaction()

                        response = chat.send_message(user_prompt)
                        print(f"Assistant: {response}\n")

                print("=" * 60)
                if session_name:
                    print(f"Conversation saved to session '{session_name}'")
                print(f"Total exchanges in this run: {len(args.prompts)}")
            elif session_name:
                # Just show the session info if no new prompts
                print(f"\nSession '{session_name}' loaded. Use --prompts to add messages.")

        else:
            # Single prompt mode
            if args.prompt:
                if monitor:
                    monitor.record_interaction()
                run_ollama_smart(selected_model, args.prompt)
    finally:
        # Stop monitoring when done
        if monitor:
            monitor.stop_monitoring()

if __name__ == "__main__":
    main()