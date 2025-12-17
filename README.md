# Hello Ollama

A Python toolkit for working with Ollama AI models, featuring interactive model selection, conversational chat, and performance benchmarking.

## Features

- **Interactive Model Chat**: Choose from installed models and chat interactively
- **Conversational Context**: Multi-turn conversations with automatic message history
- **Model Benchmarking**: Compare response times and resource usage across models
- **Automatic Installation**: Download and install models as needed
- **CSV Export**: Export benchmark results for analysis

## Prerequisites

- **Python 3.7+**: [Check your version](https://www.python.org/downloads/)
- **pip**: Usually included with Python
- **Ollama**: Download from [https://ollama.com/](https://ollama.com/)

## Installation

1. Clone and setup:
```bash
git clone <repository-url>
cd hello-ollama
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Chat with a Model

Run the main script with a model, prompt, and timeout:
```bash
python main.py "What is Python?" -m llama2 -t 5
```

**Arguments:**
- `prompt` (positional): The prompt to send to the model
- `-m, --model` (required): Model name to use
- `-t, --timeout` (required): Inactivity timeout in minutes

**Options:**
- `-i, --interactive`: Run in interactive mode (continuous prompting)
- `--skip-server-check`: Skip checking if Ollama server is running
- `--skip-model-check`: Skip checking if model is installed
- `--no-inactivity-monitor`: Disable inactivity monitoring

**Examples:**
```bash
# Single prompt with timeout
python main.py "Hello" -m llama2 -t 5

# Interactive mode (continuous prompting)
python main.py -i -m mistral -t 10

# Skip model check with custom timeout
python main.py "Tell me a joke" -m phi -t 15 --skip-model-check
```

### Benchmark Models

Compare multiple models with the same prompt:

```bash
# Benchmark all installed models
python benchmark_models.py "Explain quantum computing"

# Benchmark specific models
python benchmark_models.py "Tell me a joke" -m llama2 mistral phi

# Save to custom file
python benchmark_models.py "What is Python?" -m llama2 -o results.csv
```

**Output:** CSV file with model name, response time, CPU/memory usage, response text, and timestamp.

### Python Code Examples

**Simple conversation:**
```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

ensure_ollama_server_running()
chat = ChatSession("llama2")

response1 = chat.send_message("What is Python?")
print(response1)

response2 = chat.send_message("Show me an example")  # Model remembers context
print(response2)
```

**Install and run a model:**
```python
from src.install_ollama_model import check_and_install_model
from src.run_ollama import run_ollama_smart

check_and_install_model("mistral")
response = run_ollama_smart("mistral", "Hello!")
print(response)
```

**Get installed models:**
```python
from src.group_models import get_installed_models

models = get_installed_models()
print(f"Installed models: {models}")
```

## Project Structure

```
hello-ollama/
├── main.py                      # CLI chat interface
├── benchmark_models.py          # Model benchmarking tool
├── src/
│   ├── __init__.py             # Package initialization
│   ├── install_ollama_model.py # Model installation
│   ├── group_models.py         # Model listing and filtering
│   └── run_ollama.py           # Core Ollama wrapper
├── examples/
│   └── hello-world.py          # Simple example script
└── README.md                    # This file
```

## Advanced Usage

For advanced examples including:
- Chat sessions with system prompts
- Resource monitoring
- Direct API calls
- Programmatic benchmarking

See [CHAT_API_GUIDE.md](CHAT_API_GUIDE.md)

## Benchmarking Notes

This tool measures response time and basic resource usage. It does not evaluate:
- Response quality or accuracy
- Model capabilities across different tasks
- Token throughput or generation speed
- Consistency across multiple runs
- GPU usage (for GPU-accelerated models)

Resource metrics are sampled periodically. CPU percentage can exceed 100% on multi-core systems. Results vary based on system resources, model size, and prompt complexity.

## Tips

- **Model Names**: Check [Ollama Library](https://ollama.com/library) for available models
- **Prompt Quality**: Better prompts lead to better outputs
- **Error Handling**: Scripts handle common issues like missing models

## Resources

- [Ollama Documentation](https://ollama.com/docs/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Feel free to submit issues or pull requests.
