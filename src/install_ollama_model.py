import subprocess

def check_ollama_model_installed(model_name):
  """
  Checks if the Ollama model is installed locally and pulls it if not.
  """
  try:
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

if __name__ == "__main__":
  if check_ollama_model_installed("deepseek-coder-v2:16b"):
    print("All good!")
  else:
    print("Something went wrong.  Check the error messages.")