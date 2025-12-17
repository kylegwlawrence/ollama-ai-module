import argparse
from src.run_ollama import run_ollama

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run a simple prompt through Ollama")
  parser.add_argument("--model", default="gemma3:270m", help="Model name to use (default: gemma3:270m)")
  parser.add_argument("--prompt", default="Hello world!", help="Prompt to send to the model (default: Hello world!)")
  args = parser.parse_args()

  run_ollama(args.model, args.prompt)