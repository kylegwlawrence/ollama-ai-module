# Ollama Chat API Implementation Guide

## Overview

The hello-ollama toolkit now supports Ollama's `/api/chat` endpoint, enabling proper multi-turn conversations with automatic context management. This means the AI model will remember previous messages in the conversation without requiring you to manage message history manually.

## Key Features

✅ **Automatic Context Management** - The model remembers previous messages across prompts
✅ **Simple API** - Just call `.send_message()` repeatedly
✅ **Message History Tracking** - Inspect full conversation history anytime
✅ **System Prompts** - Customize model behavior for your use case
✅ **Resource Monitoring** - Track CPU/memory usage during conversations
✅ **No New Dependencies** - Uses existing `requests` library

## Quick Start

### Basic Conversation Loop

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

# Ensure Ollama server is running
ensure_ollama_server_running()

# Create a chat session
chat = ChatSession("llama2")

# Have a multi-turn conversation
response1 = chat.send_message("What is Python?")
print(response1)

response2 = chat.send_message("Can you show me an example?")  # Remembers context!
print(response2)

response3 = chat.send_message("Now explain what that code does")  # Still remembers!
print(response3)
```

### Programmatic Conversation Loop

Pass multiple prompts programmatically instead of typing them interactively:

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

ensure_ollama_server_running()
chat = ChatSession("mistral")

# Define conversation prompts
prompts = [
    "What is machine learning?",
    "Can you give me a simple example?",
    "How would I implement that in Python?",
    "What are some real-world applications?"
]

# Process prompts while maintaining context
for prompt in prompts:
    response = chat.send_message(prompt)
    print(f"User: {prompt}")
    print(f"Assistant: {response}\n")
```

### Using System Prompts

Customize the model's behavior:

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

ensure_ollama_server_running()
chat = ChatSession("llama2")

# Set system prompt to define behavior
chat.set_system_prompt("You are a helpful Python programming tutor. Explain concepts clearly and provide code examples.")

# Now all responses will follow this instruction
response = chat.send_message("What is a decorator?")
print(response)
```

### Monitoring Resource Usage

Track CPU and memory usage during conversations:

```python
from src.run_ollama import ChatSession, run_chat_with_monitoring, ensure_ollama_server_running

ensure_ollama_server_running()
chat = ChatSession("llama2")

# Get response with resource metrics
response, metrics = run_chat_with_monitoring(chat, "Explain quantum computing")

print(f"Response: {response}")
if metrics:
    print(f"CPU average: {metrics['cpu_avg_percent']}%")
    print(f"CPU peak: {metrics['cpu_peak_percent']}%")
    print(f"Memory average: {metrics['memory_avg_mb']} MB")
    print(f"Memory peak: {metrics['memory_peak_mb']} MB")
```

## ChatSession API Reference

### Creating a Session

```python
chat = ChatSession(model_name, host='127.0.0.1', port=11434)
```

**Parameters:**
- `model_name` (str): Name of the Ollama model (e.g., 'llama2', 'mistral')
- `host` (str, optional): Server hostname (default: '127.0.0.1')
- `port` (int, optional): Server port (default: 11434)

### Methods

#### `send_message(user_message)`
Send a message and get a response. Automatically maintains conversation context.

```python
response = chat.send_message("What is Python?")
print(response)
```

**Parameters:**
- `user_message` (str): The message text to send

**Returns:**
- (str): The assistant's response

#### `set_system_prompt(system_prompt)`
Set or update the system prompt. This affects all subsequent messages.

```python
chat.set_system_prompt("You are a helpful coding assistant.")
```

**Parameters:**
- `system_prompt` (str): System instruction text

**Note:** If a system prompt was previously set, it will be replaced. System prompts are kept separate from conversation messages.

#### `get_history()`
Get the full conversation history.

```python
history = chat.get_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

**Returns:**
- (list): List of message dictionaries with 'role' and 'content' keys

#### `reset()`
Clear all conversation history. Useful when starting a new topic.

```python
chat.reset()
# Now send_message() will start fresh
response = chat.send_message("New topic: What is AI?")
```

## Advanced Functions

### Direct API Call

For advanced use cases, you can directly use the chat endpoint API:

```python
from src.run_ollama import send_chat_message

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "Show me an example"}
]

response = send_chat_message("llama2", messages)
```

### Smart Chat Function

Automatically ensures server is running before sending messages:

```python
from src.run_ollama import run_ollama_chat_smart, ChatSession

chat = ChatSession("llama2")
response = run_ollama_chat_smart(chat, "Hello!")
```

### Chat with Monitoring

Combines chat functionality with resource tracking:

```python
from src.run_ollama import run_chat_with_monitoring, ChatSession

chat = ChatSession("llama2")
response, metrics = run_chat_with_monitoring(chat, "Tell me about AI")

# Access metrics
if metrics:
    avg_cpu = metrics['cpu_avg_percent']
    peak_memory = metrics['memory_peak_mb']
```

## Examples

### Run the Example Script

```bash
python examples/chat_conversation.py
```

This demonstrates:
- Creating a chat session
- Setting system prompts
- Running a multi-turn conversation
- Tracking conversation statistics
- Resetting for a new topic

### Example: Conversation Loop from File

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

ensure_ollama_server_running()

# Read prompts from a file
with open('prompts.txt', 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

chat = ChatSession("llama2")
for prompt in prompts:
    response = chat.send_message(prompt)
    print(f"Q: {prompt}\nA: {response}\n")
    print("-" * 60)
```

### Example: Building an Interactive Bot

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running

ensure_ollama_server_running()

chat = ChatSession("mistral")
chat.set_system_prompt("You are a helpful assistant for learning Python.")

print("Chat Bot (type 'quit' to exit)")
print("-" * 40)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        break

    try:
        response = chat.send_message(user_input)
        print(f"Bot: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

print("Conversation ended.")
```

## How It Works

### The Chat Endpoint Difference

**Traditional `/api/generate` (stateless):**
- Each prompt is independent
- No context from previous prompts
- Fast but requires manual history management

**New `/api/chat` (stateful):**
- Accepts message history
- Model reads entire conversation
- Automatically maintains context
- Perfect for multi-turn conversations

### Message Format

Chat messages follow this structure:

```python
{
    "role": "user",      # or "assistant" or "system"
    "content": "text"    # The message text
}
```

**Roles:**
- `"system"`: System instructions (optional, defines model behavior)
- `"user"`: User's message
- `"assistant"`: Model's response

### ChatSession Internals

The ChatSession class:
1. Stores all messages in a `messages` list
2. When you call `send_message()`, it:
   - Adds your message with role="user"
   - Sends entire history to `/api/chat`
   - Gets response and adds it with role="assistant"
   - Returns just the response text
3. History persists until you call `reset()`

## Tips & Best Practices

### 1. Set Meaningful System Prompts

Different system prompts make the model behave differently:

```python
# For code assistance
chat.set_system_prompt("You are an expert Python programmer. Provide clear code examples.")

# For creative writing
chat.set_system_prompt("You are a creative fiction writer.")

# For tutoring
chat.set_system_prompt("You are a patient tutor. Explain complex topics simply.")
```

### 2. Reset Between Topics

Start fresh when switching topics:

```python
# Conversation about Python
chat.send_message("Explain decorators")

# Reset for different topic
chat.reset()
chat.set_system_prompt("You are a music theory expert.")
chat.send_message("What is a chord progression?")
```

### 3. Handle Errors Gracefully

```python
try:
    response = chat.send_message("Some prompt")
except Exception as e:
    print(f"Error communicating with Ollama: {e}")
    # Optionally reset and retry, or inform user
```

### 4. Monitor Long Conversations

For long conversations, message history grows. Monitor or limit if needed:

```python
history = chat.get_history()
message_count = len(history)
print(f"Conversation has {message_count} messages")

if message_count > 100:
    print("Warning: Large conversation history may affect performance")
```

### 5. Combine with Other Features

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running, InactivityMonitor

ensure_ollama_server_running()

chat = ChatSession("llama2")
monitor = InactivityMonitor("llama2", timeout_minutes=5)
monitor.start_monitoring()

try:
    # Your conversation here
    response = chat.send_message("Hello")
finally:
    monitor.stop_monitoring()
```

## Integration with Existing Features

### With Resource Monitoring

```python
from src.run_ollama import ChatSession, run_chat_with_monitoring

chat = ChatSession("llama2")
response, metrics = run_chat_with_monitoring(chat, "Tell me about AI")
```

### With Server Management

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running, stop_ollama_model

ensure_ollama_server_running()

chat = ChatSession("llama2")
chat.send_message("Hello")

# Clean up when done
stop_ollama_model("llama2")
```

### With Model Installation

```python
from src.run_ollama import ChatSession, ensure_ollama_server_running
from src.install_ollama_model import check_and_install_model

ensure_ollama_server_running()
check_and_install_model("llama2")

chat = ChatSession("llama2")
response = chat.send_message("Hello")
```

## Comparison: Before vs After

### Before (Stateless)
```python
# Each prompt is independent - no context
response1 = run_ollama_smart("llama2", "What is Python?")
response2 = run_ollama_smart("llama2", "Can you show me an example?")  # Doesn't know previous context
```

### After (Stateful)
```python
# Full conversation context maintained
chat = ChatSession("llama2")
response1 = chat.send_message("What is Python?")
response2 = chat.send_message("Can you show me an example?")  # Knows we're discussing Python
```

## Troubleshooting

### "Connection refused" errors
- Ensure Ollama server is running: `ollama serve`
- Check that server is on correct host/port
- Verify firewall isn't blocking port 11434

### Model not responding
- Verify model is installed: `ollama list`
- Pull model if needed: `ollama pull llama2`
- Check server logs for errors

### Large conversation memory usage
- Call `chat.reset()` periodically
- Monitor history size with `len(chat.get_history())`

### Slow responses with long history
- Long message histories take more time to process
- Consider resetting periodically for better performance

## Next Steps

1. **Try the example**: `python examples/chat_conversation.py`
2. **Modify for your use case**: Edit the prompts or system prompt
3. **Build your application**: Use ChatSession as foundation
4. **Explore advanced features**: Combine with monitoring and server management

## API Reference Summary

| Function | Purpose |
|----------|---------|
| `ChatSession(model_name)` | Create a chat session |
| `.send_message(text)` | Send message and get response |
| `.set_system_prompt(text)` | Set behavior instructions |
| `.get_history()` | Get all messages |
| `.reset()` | Clear conversation |
| `send_chat_message(model, messages)` | Direct API call |
| `run_ollama_chat_smart(chat, message)` | Chat with auto server check |
| `run_chat_with_monitoring(chat, message)` | Chat with resource tracking |

## Summary

The chat endpoint implementation provides:
- ✅ True multi-turn conversations with context
- ✅ Programmatic conversation loops (no terminal input needed)
- ✅ Simple, intuitive API
- ✅ Full history tracking and control
- ✅ System prompt customization
- ✅ Integration with existing features
- ✅ No new dependencies required

You can now build sophisticated conversational applications with Ollama!
