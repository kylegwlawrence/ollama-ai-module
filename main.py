from src.select_model import ask_model_selection
from src.install_ollama_model import check_ollama_model_installed
from src.run_ollama import run_ollama

def main():
    # Get the selected model from the user
    selected_model = ask_model_selection()

    # Check if the model is installed
    check_ollama_model_installed(selected_model)

    # Get the prompt from the user
    user_prompt = input("Enter your prompt: ")

    # Run Ollama with the selected model and prompt
    run_ollama(selected_model, user_prompt)

if __name__ == "__main__":
    main()