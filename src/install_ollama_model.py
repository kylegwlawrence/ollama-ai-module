import subprocess
from src.run_ollama import ensure_ollama_running

def check_and_install_model(model_name):
  """
  Checks if the Ollama model is installed locally and pulls it if not.
  """
  try:
    # Ensure the Ollama server is running before attempting to pull models
    ensure_ollama_running()

    command = ["ollama", "--version"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    if "ollama version" in result.stdout:
      print(f"Ollama is installed: {result.stdout.strip()}")
      print(f"Checking if model '{model_name}' is available...")

      # Check if model is already pulled
      list_command = ["ollama", "list"]
      list_result = subprocess.run(list_command, capture_output=True, text=True, check=True)

      if model_name in list_result.stdout:
        print(f"Model '{model_name}' is already installed.")
        return True
      else:
        print(f"Model '{model_name}' not found. Pulling...")
        pull_command = ["ollama", "pull", model_name]
        subprocess.run(pull_command, capture_output=True, text=True, check=True)
        print(f"Model '{model_name}' successfully pulled.")
        return True
    else:
      print("Unexpected Ollama version output.")
      return False

  except subprocess.CalledProcessError:
    print("Ollama not found.  Please ensure Ollama is installed and in your PATH.")
    return False

def check_and_install_models(models):
  """
  Checks and installs multiple Ollama models from a list.

  Args:
    models: A list of model names to check and install.

  Returns:
    A list of tuples (model_name, success) indicating which models were successfully installed.
  """
  results = []
  for model in models:
    success = check_and_install_model(model)
    results.append((model, success))
  return results

if __name__ == "__main__":
  if check_and_install_model("deepseek-coder-v2:16b"):
    print("All good!")
  else:
    print("Something went wrong.  Check the error messages.")