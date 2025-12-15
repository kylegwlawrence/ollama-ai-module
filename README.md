# Hello Ollama

A Python toolkit for working with Ollama AI models, featuring interactive model selection and performance benchmarking capabilities.

## Features

- **Interactive Model Selection**: Choose from locally installed Ollama models with an intuitive CLI interface
- **Model Benchmarking**: Compare response times and outputs across multiple AI models
- **Automatic Model Installation**: Automatically downloads and installs models if not already available
- **CSV Export**: Export benchmark results for analysis and comparison

## Prerequisites

* **Python:** Python 3.7 or higher. Check your version:
  ```bash
  python --version
  # or
  python3 --version
  ```

* **pip:** Python's package installer (usually included with Python). Update it:
  ```bash
  pip install --upgrade pip
  ```

* **Ollama:** Download and install from [https://ollama.com/](https://ollama.com/)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd hello-ollama
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

## Usage

### Interactive Model Chat

Run the main script to select a model and chat interactively:

```bash
python main.py
```

This will:
1. Show available installed models
2. Let you select a model or enter a custom model name
3. Check if the model is installed (and install if needed)
4. Prompt you for input to send to the model

### Benchmarking Models

Compare multiple models with the same prompt:

```bash
# Benchmark all installed models
python benchmark_models.py "Explain quantum computing in simple terms"

# Benchmark specific models
python benchmark_models.py "Tell me a joke" -m llama2 mistral phi

# Specify custom output file
python benchmark_models.py "What is Python?" -m llama2 -o results.csv
```

**Options:**
- `prompt` (required): The prompt to test all models with
- `-m, --models`: List of model names (if not provided, uses all installed models)
- `-o, --output`: Output CSV filename (default: model_benchmark.csv)

**Output:**
The benchmark creates a CSV file with:
- `model_name`: Name of the AI model
- `duration_seconds`: Response time in seconds
- `response`: The full AI response
- `timestamp`: When the prompt was sent

**Important Note on Benchmarking:**
This is a simplified benchmarking tool that measures only response time for a single prompt. It does not evaluate:
- Response quality or accuracy
- Model capabilities across different task types
- Token throughput or generation speed
- Memory usage or system resources
- Consistency across multiple runs

Response time can vary significantly based on system resources, model size, prompt complexity, and other factors. This tool is intended for basic comparison purposes only and should not be considered a comprehensive performance evaluation.

## Project Structure

```
hello-ollama/
├── benchmark_models.py          # Model benchmarking tool
├── main.py                      # Interactive chat interface
├── src/
│   ├── install_ollama_model.py # Model installation utilities
│   ├── run_ollama.py           # Core Ollama execution wrapper
│   └── select_model.py         # Model selection interface
├── examples/
│   └── hello-world.py          # Simple example script
└── README.md                    # This file
```

## Basic Scripting Example

Here's a simple Python script that demonstrates how to run a language model with Ollama:

```python
import subprocess

def run_ollama(model_name, prompt):
  """Runs an Ollama model and returns the output."""
  try:
    result = subprocess.run(['ollama', 'run', model_name],
                          input=prompt, capture_output=True, text=True, check=True)
    print(result.stdout)
  except subprocess.CalledProcessError as e:
    print(f"Error running Ollama: {e}")

if __name__ == "__main__":
  model_name = "llama2"
  prompt = "Hello, how are you today?"
  run_ollama(model_name, prompt)
```

### Explanation

* **`import subprocess`:** Essential for running external commands like Ollama
* **`run_ollama(model_name, prompt)`:** Encapsulates the logic for running the model
  * `subprocess.run(...)`: Executes the Ollama command
  * `capture_output=True`: Captures stdout and stderr
  * `text=True`: Decodes output as text
  * `check=True`: Raises exception on non-zero exit code
* **`try...except` block:** Handles potential errors gracefully

## Advanced Usage

### Using the Module Functions

```python
from src.select_model import get_installed_models
from src.install_ollama_model import check_ollama_model_installed
from src.run_ollama import run_ollama

# Get list of installed models
models = get_installed_models()
print(f"Installed models: {models}")

# Ensure a model is installed
check_ollama_model_installed("llama2")

# Run a model and get output
response = run_ollama("llama2", "Tell me a joke", return_output=True)
print(response)
```

### Programmatic Benchmarking

```python
from benchmark_models import benchmark_models

# Benchmark specific models
benchmark_models(
    models=['llama2', 'mistral'],
    prompt="What is artificial intelligence?",
    output_file="my_benchmark.csv"
)

# Benchmark all installed models
benchmark_models(
    prompt="Explain machine learning",
    use_installed=True,
    output_file="all_models.csv"
)
```

## Important Considerations

* **Model Names:** Check the [Ollama model library](https://ollama.com/library) for available models
* **Prompt Engineering:** The quality of your prompt significantly impacts the model's output
* **Output Handling:** Model outputs are returned via stdout and can be parsed for complex tasks
* **Error Handling:** The scripts include error handling for common issues like missing models or network problems

## Resources

* **Ollama Documentation:** [https://ollama.com/docs/](https://ollama.com/docs/)
* **Ollama GitHub Repository:** [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
* **Ollama Model Library:** [https://ollama.com/library](https://ollama.com/library)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
