from src.run_ollama import run_ollama

if __name__ == "__main__":
  model_name = "gemma3:270m"
  prompt = "Hello world!"
  run_ollama(model_name, prompt)