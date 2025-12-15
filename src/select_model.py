import subprocess
import sys


def get_installed_models():
    """
    Get list of locally installed Ollama models using 'ollama list' command.

    Returns:
        list: List of model names
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output - skip the header line
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return []

        # Extract model names (first column)
        models = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                # Model name is the first column
                model_name = line.split()[0]
                models.append(model_name)

        return models

    except subprocess.CalledProcessError as e:
        print(f"Error running 'ollama list': {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed.", file=sys.stderr)
        return []


def ask_model_selection():
    """
    Prompt user to select an Ollama model from locally installed models,
    or enter a custom model name.

    Returns:
        str: Selected model name, or None if user cancels
    """
    models = get_installed_models()

    if not models:
        print("No Ollama models found locally.")
        custom = input("Enter a model name to use (or 'q' to quit): ").strip()
        if custom.lower() == 'q':
            return None
        return custom if custom else None

    print("\nAvailable Ollama models:")
    print("-" * 40)
    for idx, model in enumerate(models, 1):
        print(f"{idx}. {model}")
    print("-" * 40)

    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(models)}), enter custom model name, or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                return None

            # Try to parse as a number first
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]
                    print(f"\nSelected model: {selected_model}")
                    return selected_model
                else:
                    print(f"Please enter a number between 1 and {len(models)}, or enter a custom model name")
            except ValueError:
                # Not a number, treat as custom model name
                if choice:
                    print(f"\nUsing custom model: {choice}")
                    return choice
                else:
                    print("Please enter a valid model name or number")

        except KeyboardInterrupt:
            print("\n\nSelection cancelled.")
            return None


if __name__ == "__main__":
    model = ask_model_selection()
    if model:
        print(f"\nYou selected: {model}")
    else:
        print("\nNo model selected.")
