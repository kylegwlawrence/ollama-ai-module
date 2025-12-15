import subprocess

def run_ollama(model_name, prompt, return_output=False):
  """Runs an Ollama model and returns the output."""
  try:
    result = subprocess.run(['ollama', 'run', model_name],
    input=prompt, capture_output=True, text=True, check=True)
    if return_output:
      return result.stdout
    else:
      print(result.stdout)
  except subprocess.CalledProcessError as e:
    if return_output:
      raise
    else:
      print(f"Error running Ollama: {e}")