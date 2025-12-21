# hello-ollama

A Python library for interacting with [Ollama](https://ollama.ai) - a local AI model runner. This library provides utilities for managing Ollama servers, running language models, and maintaining persistent chat sessions.

## Features

- **Interactive CLI**: Full-featured terminal interface with conversation management
- **Chat Sessions**: Start and continue multi-turn conversations with persistent history stored as JSON
- **Conversation Management**: Resume, view history, and delete past conversations
- **Model Management**: Automatically detect installed models and manage model execution
- **Resource Monitoring**: Track CPU and memory usage during model inference
- **Flexible Execution**: Use Ollama's REST API or CLI interface automatically
- **Type Safe**: Full type hints throughout the codebase

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and available on your system

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd hello-ollama
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive CLI

Run the interactive chat interface:

```bash
python ollama_chat.py
```

Features:
- **Start a new chat**: Choose a model and begin a fresh conversation
- **Resume existing chat**: Continue from where you left off with full conversation history
- **Delete conversations**: Remove old conversations you no longer need
- **Automatic summarization**: Conversations are summarized for easy identification

### Using as a Library

#### Starting a New Chat Session

```python
from ollama_chat import start_chat_session

result = start_chat_session(
    prompt="Hello, how are you?",
    model_name="smollm2:135m",
    system_prompt="You are a helpful assistant"
)

print(result['response'])  # Model's response
print(result['session_name'])  # Session identifier for future conversations
```

#### Continuing a Chat Session

```python
from ollama_chat import continue_chat_session

result = continue_chat_session(
    prompt="Tell me more about that",
    session_name="happy-blue-penguin"  # From previous session
)

print(result['response'])
```

## Core Components

### OllamaServer

Manages the connection to your local Ollama server and validates availability.

```python
from src.server import OllamaServer

server = OllamaServer()
models = server.list_models()
```

### OllamaModel

Executes prompts using a specified model. Automatically selects between REST API and CLI execution.

```python
from src.model import OllamaModel
from src.server import OllamaServer

server = OllamaServer()
model = OllamaModel("smollm2:135m", server)
response = model.send_prompt("Your prompt here", return_output=True)
```

### ChatSession

Manages multi-turn conversations with persistent storage. Sessions are saved as JSON files in `.conversations/` directory.

```python
from src.chat_session import ChatSession
from src.model import OllamaModel
from src.server import OllamaServer

server = OllamaServer()
model = OllamaModel("smollm2:135m", server)
session = ChatSession(model, "my-session-name")

# Send a message
response = session.send_message("Hello!")

# Set system prompt
session.set_system_prompt("You are a helpful expert")

# Get conversation history
history = session.get_history()

# List all sessions
all_sessions = ChatSession.list_sessions()

# Delete a session
ChatSession.delete_session("my-session-name")
```

### ResourceMonitor

Tracks CPU and memory usage of a process in real-time.

```python
from src.resource_monitor import ResourceMonitor
import os

monitor = ResourceMonitor(os.getpid())
monitor.start()
# ... do some work ...
monitor.stop()

print(f"Avg CPU: {monitor.get_average_cpu()}%")
print(f"Peak Memory: {monitor.get_peak_memory()}MB")
```

### OllamaAPIClient

Low-level HTTP client for direct API communication with Ollama server.

```python
from src.api_client import OllamaAPIClient

client = OllamaAPIClient()
models = client.list_models()
response = client.generate("smollm2:135m", "Your prompt here")

# Default timeout increased to 600s for long-running generations
response = client.chat("llama2", messages, timeout=600)
```

## API Reference

### start_chat_session(prompt, model_name, system_prompt)

Start a new conversation with an Ollama model.

**Parameters:**
- `prompt` (str): The initial user prompt
- `model_name` (str): Name of the model (e.g., 'smollm2:135m', 'llama2')
- `system_prompt` (str): System instructions for the model

**Returns:** Dictionary with:
- `response` (str): Model's response to the initial prompt
- `session_name` (str): Unique session identifier

### continue_chat_session(prompt, session_name)

Continue an existing conversation.

**Parameters:**
- `prompt` (str): User's message in the ongoing conversation
- `session_name` (str): Session identifier from a previous session

**Returns:** Dictionary with:
- `response` (str): Model's response
- `session_name` (str): Session identifier

## Project Structure

```
hello-ollama/
├── src/
│   ├── api_client.py          # Ollama HTTP API client
│   ├── server.py              # Server management and model listing
│   ├── model.py               # Model execution and monitoring
│   ├── chat_session.py        # Chat session persistence
│   ├── resource_monitor.py    # CPU/memory monitoring
│   ├── utilities.py           # Helper functions
│   └── __init__.py
├── examples/
│   ├── start_chat_session.py  # Example: starting a new session
│   ├── continue_chat_session.py # Example: continuing a session
│   ├── server.py              # Example: server interaction
│   └── resource_monitor.py    # Example: monitoring resources
├── .conversations/            # Stored chat sessions (auto-created)
├── ollama_chat.py             # Interactive CLI and public API
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Dependencies

- **psutil** (>=5.11.0): Process and system monitoring
- **requests** (>=2.31.0): HTTP client for Ollama API
- **human_id** (>=0.2.0): Generate human-readable session IDs

## Session Storage

Chat sessions are automatically saved to `.conversations/` directory as JSON files. Each session file contains:

```json
{
  "session_name": "happy-blue-penguin",
  "model": "smollm2:135m",
  "created_at": "2024-01-15T10:30:45.123456",
  "last_updated": "2024-01-15T10:35:20.654321",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

## Examples

See the `examples/` directory for complete working examples:

- [Start Chat Session](examples/start_chat_session.py)
- [Continue Chat Session](examples/continue_chat_session.py)
- [Server Interaction](examples/server.py)
- [Resource Monitoring](examples/resource_monitor.py)

## Error Handling

The library provides custom exceptions for different error scenarios:

```python
from src.api_client import (
    OllamaAPIException,
    OllamaConnectionError,
    OllamaTimeoutError
)

try:
    result = start_chat_session(prompt, model, system_prompt)
except OllamaConnectionError:
    print("Could not connect to Ollama server")
except OllamaTimeoutError:
    print("Request timed out")
except OllamaAPIException as e:
    print(f"API error: {e}")
```

## Notes

- Ollama server must be running before using this library
- Models are lazy-loaded on first use
- Session persistence is automatic with each message sent
- Resource monitoring runs in a background thread

## Contributing

Contributions are welcome! Please ensure all code includes type hints and follows the existing code style.
