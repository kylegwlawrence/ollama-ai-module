import argparse
from src.select_model import ask_model_selection
from src.install_ollama_model import check_and_install_model
from src.run_ollama import run_ollama_smart, ensure_ollama_server_running, InactivityMonitor

# Default inactivity timeout in minutes
DEFAULT_INACTIVITY_TIMEOUT = 5

def main():
    parser = argparse.ArgumentParser(
        description='Run Ollama models with optional server management and inactivity monitoring'
    )

    # Model selection arguments
    parser.add_argument(
        'prompt',
        nargs='?',
        default=None,
        help='The prompt to send to the model'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Model name to use (if not provided, will prompt for selection)'
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
        default=DEFAULT_INACTIVITY_TIMEOUT,
        help=f'Inactivity timeout in minutes (default: {DEFAULT_INACTIVITY_TIMEOUT})'
    )
    parser.add_argument(
        '--no-inactivity-monitor',
        action='store_true',
        help='Disable inactivity monitoring'
    )

    # Interactive mode
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (continuous prompting until quit)'
    )

    args = parser.parse_args()

    # Check if prompt is required
    if not args.prompt and not args.interactive and not args.model:
        parser.print_help()
        return

    # Ensure Ollama server is running
    if not args.skip_server_check:
        ensure_ollama_server_running()

    # Get or select model
    if args.model:
        selected_model = args.model
    else:
        selected_model = ask_model_selection()

    # Check if model is installed
    if not args.skip_model_check:
        check_and_install_model(selected_model)

    # Start inactivity monitor if not disabled
    monitor = None
    if not args.no_inactivity_monitor and args.timeout > 0:
        monitor = InactivityMonitor(selected_model, args.timeout)
        monitor.start_monitoring()

    try:
        if args.interactive:
            # Interactive mode: continuous prompting
            while True:
                try:
                    user_prompt = input("Enter your prompt (or 'quit' to exit): ")

                    if user_prompt.lower() == 'quit':
                        break

                    if user_prompt.strip():
                        if monitor:
                            monitor.record_interaction()
                        run_ollama_smart(selected_model, user_prompt)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
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