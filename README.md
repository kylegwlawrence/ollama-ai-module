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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
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

### Conversational Chat with Context

Have multi-turn conversations where the model remembers previous messages:

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

# Ensure server is running
ensure_ollama_server_running()

# Create chat session
chat = ChatSession("llama2")

# Optional: Set system prompt
chat.set_system_prompt("You are a helpful coding assistant.")

# Have a conversation
response1 = chat.send_message("What is Python?")
print(response1)

response2 = chat.send_message("Can you show me an example?")  # Model remembers context
print(response2)

response3 = chat.send_message("Now explain that code")  # Model still remembers
print(response3)

# Reset conversation if needed
chat.reset()
```

**Key Features:**
- Automatic context management - model remembers conversation history
- Simple API - just call `send_message()` repeatedly
- Reset capability to start fresh conversations
- System prompt support for behavior customization

### Benchmarking Models

Compare multiple models with the same prompt:

```bash
# Benchmark all installed models
python -m src.benchmark_models "Explain quantum computing in simple terms"

# Benchmark specific models
python -m src.benchmark_models "Tell me a joke" -m llama2 mistral phi

# Specify custom output file
python -m src.benchmark_models "What is Python?" -m llama2 -o results.csv
```

**Options:**
- `prompt` (required): The prompt to test all models with
- `-m, --models`: List of model names (if not provided, uses all installed models)
- `-o, --output`: Output CSV filename (default: model_benchmark.csv)

**Output:**
The benchmark creates a CSV file with:
- `model_name`: Name of the AI model
- `duration_seconds`: Response time in seconds
- `cpu_avg_percent`: Average CPU usage during execution
- `cpu_peak_percent`: Peak CPU usage during execution
- `memory_avg_mb`: Average memory usage in MB
- `memory_peak_mb`: Peak memory usage in MB
- `response`: The full AI response
- `timestamp`: When the prompt was sent

**Important Note on Benchmarking:**
This is a simplified benchmarking tool that measures response time and basic resource usage (CPU and memory) for a single prompt. It does not evaluate:
- Response quality or accuracy
- Model capabilities across different task types
- Token throughput or generation speed
- Consistency across multiple runs
- Disk I/O or network usage
- GPU usage (for GPU-accelerated models)

Resource metrics are sampled every 0.5 seconds. CPU percentage can exceed 100% on multi-core systems (e.g., 200% = 2 full cores). Response time and resource usage can vary significantly based on system resources, model size, prompt complexity, and other factors. This tool is intended for basic comparison purposes only and should not be considered a comprehensive performance evaluation.

## Project Structure

```
hello-ollama/
├── main.py                      # Interactive chat interface
├── src/
│   ├── __init__.py             # Package initialization
│   ├── __main__.py             # Entry point for module execution
│   ├── benchmark_models.py     # Model benchmarking tool
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
from src.install_ollama_model import check_and_install_model
from src.run_ollama import run_ollama, ChatSession, send_chat_message, run_chat_with_monitoring

# Get list of installed models
models = get_installed_models()
print(f"Installed models: {models}")

# Ensure a model is installed
check_and_install_model("llama2")

# Run a model and get output
response = run_ollama("llama2", "Tell me a joke", return_output=True)
print(response)

# Create and use a chat session
chat = ChatSession("llama2", host="127.0.0.1", port=11434)
response = chat.send_message("Hello!")
print(response)

# Direct API call for advanced use cases
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "Show me an example"}
]
response = send_chat_message("llama2", messages)
print(response)
```

### Using Chat Sessions for Conversations

Use ChatSession for multi-turn conversations with automatic context management:

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

ensure_ollama_server_running()

# Create a chat session
chat = ChatSession("mistral")

# Have a multi-turn conversation
prompts = [
    "What is machine learning?",
    "Can you give me a simple example?",
    "How would I implement that in Python?"
]

for prompt in prompts:
    response = chat.send_message(prompt)
    print(f"User: {prompt}")
    print(f"Assistant: {response}\n")

# Inspect conversation history
history = chat.get_history()
print(f"Conversation had {len(history)} messages")
```

**Benefits:**
- Model automatically remembers previous messages
- Perfect for building intelligent applications
- Simple API - no manual history management needed

### Chat with Resource Monitoring

Monitor system resources while having conversations:

```python
from src.run_ollama import ChatSession, run_chat_with_monitoring, ensure_ollama_server_running

ensure_ollama_server_running()
chat = ChatSession("llama2")

# Get response with resource metrics
response, metrics = run_chat_with_monitoring(chat, "Tell me about AI")
print(f"Response: {response}")
if metrics:
    print(f"CPU avg: {metrics['cpu_avg_percent']}%")
    print(f"Memory avg: {metrics['memory_avg_mb']} MB")
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
