import subprocess
from src.run_ollama import ensure_ollama_server_running

def is_model_installed(model_name: str) -> bool:
  """
  Checks if the Ollama model is installed locally.
  """
  try:
    command = ["ollama", "--version"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    if "ollama version" not in result.stdout:
      print("Unexpected Ollama version output.")
      return False

    print(f"Ollama is installed: {result.stdout.strip()}")
    print(f"Checking if model '{model_name}' is available...")

    list_command = ["ollama", "list"]
    list_result = subprocess.run(list_command, capture_output=True, text=True, check=True)

    if model_name in list_result.stdout:
      print(f"Model '{model_name}' is already installed.")
      return True
    else:
      print(f"Model '{model_name}' not found.")
      return False

  except subprocess.CalledProcessError:
    print("Ollama not found.  Please ensure Ollama is installed and in your PATH.")
    return False

def install_model(model_name: str) -> bool:
  """
  Installs the Ollama model by pulling it.
  """
  try:
    print(f"Pulling model '{model_name}'...")
    pull_command = ["ollama", "pull", model_name]
    subprocess.run(pull_command, capture_output=True, text=True, check=True)
    print(f"Model '{model_name}' successfully pulled.")
    return True

  except subprocess.CalledProcessError:
    print(f"Failed to pull model '{model_name}'.")
    return False

def check_and_install_model(model_name: str) -> bool:
  """
  Checks if the Ollama model is installed locally and pulls it if not.
  """
  # Ensure the Ollama server is running before attempting to pull models
  ensure_ollama_server_running()

  if is_model_installed(model_name):
    return True
  else:
    return install_model(model_name)

if __name__ == "__main__":
  if check_and_install_model("deepseek-coder-v2:16b"):
    print("All good!")
  else:
    print("Something went wrong.  Check the error messages.")