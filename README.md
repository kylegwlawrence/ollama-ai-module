# hello-ollama

A lightweight Python library and CLI for interacting with [Ollama](https://ollama.ai) - a local AI model runner. Manage persistent chat sessions, monitor resources, and interact with language models through a clean API.

## Features

- **Interactive CLI**: Full-featured terminal interface with conversation management and navigation
- **Persistent Chats**: Multi-turn conversations with automatic JSON storage and history
- **Auto-Summarization**: Conversations summarized on exit for easy identification
- **Flexible Navigation**: Return to main menu from any submenu with 'back' command
- **Model Management**: Auto-detect installed models with grouped filtering
- **Resource Monitoring**: Real-time CPU and memory tracking
- **Dual Execution**: Automatic fallback between REST API and CLI
- **Type Safe**: Full type hints throughout

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running locally

### Setup

```bash
git clone <repository-url>
cd hello-ollama
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Optional - For Conversation Summarization:**

Pull the summarization model (used to auto-summarize chats on exit):
```bash
ollama pull gemma3:1b-it-q4_K_M
```

Or customize by changing `CHAT_SUMMARY_MODEL` constant in `src/chat_session.py` (line 7) to any model you prefer.

## Quick Start

### Interactive CLI

```bash
python ollama_chat.py
```

**Menu Options:**
1. **Start new chat** - Select model and begin fresh conversation
2. **Resume chat** - Continue existing conversation with full history (type 'back' to return)
3. **Delete conversation** - Remove conversations with confirmation (type 'back' to return)

**During Chat:**
- Type messages to converse with the AI
- Type `exit` or `quit` to end (auto-summarizes conversation)
- Press Ctrl+C for graceful exit

### Using as a Library

```python
from ollama_chat import start_chat_session, continue_chat_session

# Start new conversation
result = start_chat_session(
    prompt="Hello, how are you?",
    model_name="smollm2:135m",
    system_prompt="You are a helpful assistant"
)
print(result['response'])
print(result['session_name'])  # e.g., "happy-blue-penguin"

# Continue conversation
result = continue_chat_session(
    prompt="Tell me more",
    session_name="happy-blue-penguin"
)
print(result['response'])
```

## Core Components

### ChatSession - Persistent Conversations

```python
from src.chat_session import ChatSession
from src.model import OllamaModel
from src.server import OllamaServer

server = OllamaServer()
model = OllamaModel("smollm2:135m", server)
session = ChatSession(model, "my-session-name")

session.set_system_prompt("You are a coding expert")
response = session.send_message("Explain Python decorators")
history = session.get_history()

# Session management
all_sessions = ChatSession.list_sessions()
session_info = ChatSession.get_session_info("my-session-name")
ChatSession.delete_session("my-session-name")
```

### OllamaServer - Server Management

```python
from src.server import OllamaServer

server = OllamaServer()
models = server.get_installed_models()
is_available = server.is_server_available()
```

### ResourceMonitor - Track Performance

```python
from src.resource_monitor import ResourceMonitor
import os

monitor = ResourceMonitor(os.getpid())
monitor.start()
# ... do work ...
monitor.stop()

print(f"Avg CPU: {monitor.get_average_cpu()}%")
print(f"Peak Memory: {monitor.get_peak_memory()}MB")
```

### OllamaAPIClient - Low-Level API

```python
from src.api_client import OllamaAPIClient

client = OllamaAPIClient()
models = client.list_models()
response = client.generate("smollm2:135m", "Your prompt")
chat_response = client.chat("llama2", messages, timeout=600)
```

## Session Storage

Sessions auto-save to `.conversations/` as JSON:

```json
{
  "session_name": "happy-blue-penguin",
  "model": "smollm2:135m",
  "created_at": "2024-01-15T10:30:45.123456",
  "last_updated": "2024-01-15T10:35:20.654321",
  "conversation_summary": "Discussion about Python decorators",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

## Project Structure

```
hello-ollama/
├── src/
│   ├── api_client.py         # HTTP client for Ollama API
│   ├── server.py             # Server connection & model listing
│   ├── model.py              # Model execution & monitoring
│   ├── chat_session.py       # Session persistence & management
│   ├── resource_monitor.py   # CPU/memory tracking
│   ├── utilities.py          # Human-readable ID generation
│   └── llama_art.py          # ASCII art for CLI
├── examples/                 # Working code examples
├── .conversations/           # Session storage (auto-created)
├── ollama_chat.py           # CLI entry point & public API
└── requirements.txt         # Dependencies
```

## Error Handling

```python
from src.api_client import (
    OllamaAPIException,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaHTTPError
)

try:
    result = start_chat_session(prompt, model, system_prompt)
except OllamaConnectionError:
    print("Cannot connect to Ollama server")
except OllamaTimeoutError:
    print("Request timed out (default: 600s)")
except OllamaHTTPError as e:
    print(f"HTTP error: {e}")
```

## Configuration

**Server Settings** (hardcoded in `src/api_client.py`):
- Host: `127.0.0.1`
- Port: `11434`
- Health Check Timeout: `2s`
- Request Timeout: `600s`

**Summarization** (in `src/chat_session.py`):
- Default model: `gemma3:1b-it-q4_K_M` (configurable via `CHAT_SUMMARY_MODEL` constant)
- Focuses on first 3-5 message exchanges
- Triggered on graceful exit
- Requires model to be pulled: `ollama pull gemma3:1b-it-q4_K_M`

## Examples

See `examples/` directory:
- `start_chat_session.py` - Start new conversation
- `continue_chat_session.py` - Resume conversation
- `server.py` - Server interaction
- `benchmark_model.py` - Model benchmarking
- `resource_monitor.py` - Resource tracking

## Dependencies

- `psutil>=5.11.0` - System resource monitoring
- `requests>=2.31.0` - HTTP client
- `human_id>=0.2.0` - Readable session IDs

## Notes

- Ollama server must be running before use
- Sessions auto-save after each message
- Session names sanitized (alphanumeric + hyphens/underscores)
- Resource monitoring runs in daemon thread
- Summary generation uses separate small model (gemma3:1b-it-q4_K_M)
