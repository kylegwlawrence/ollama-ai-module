from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

from .api_client import OllamaAPIClient


# Built-in tool definitions for file operations
FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the local filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file on the local filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    }
]


def _execute_read_file(path: str, base_path: Optional[Path] = None) -> str:
    """Execute read_file tool."""
    file_path = Path(path)
    if base_path and not file_path.is_absolute():
        file_path = base_path / file_path

    if not file_path.exists():
        return f"Error: File '{file_path}' not found"

    try:
        return file_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


def _execute_write_file(path: str, content: str, base_path: Optional[Path] = None) -> str:
    """Execute write_file tool."""
    file_path = Path(path)
    if base_path and not file_path.is_absolute():
        file_path = base_path / file_path

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing file: {e}"


TOOL_EXECUTORS: Dict[str, Callable[..., str]] = {
    "read_file": _execute_read_file,
    "write_file": _execute_write_file,
}


class ToolAgent:
    """Agent that runs a chat loop with tool execution."""

    def __init__(self, client: OllamaAPIClient, model: str,
                 tools: Optional[List[Dict]] = None,
                 tool_executors: Optional[Dict[str, Callable]] = None,
                 base_path: Optional[Path] = None,
                 max_iterations: int = 10):
        self.client = client
        self.model = model
        self.tools = tools or FILE_TOOLS
        self.tool_executors = tool_executors or TOOL_EXECUTORS
        self.base_path = base_path or Path.cwd()
        self.max_iterations = max_iterations

    def run(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent with a prompt, executing tools as needed.

        Returns:
            Dict with 'response' (final text), 'messages' (full history),
            and 'tool_calls' (list of all tool calls made)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        all_tool_calls = []

        for _ in range(self.max_iterations):
            response = self.client.chat_with_tools(
                model=self.model,
                messages=messages,
                tools=self.tools
            )

            assistant_message = response.get("message", {})
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                # No more tool calls, return final response
                return {
                    "response": assistant_message.get("content", ""),
                    "messages": messages,
                    "tool_calls": all_tool_calls
                }

            # Execute each tool call
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name")
                args = func.get("arguments", {})

                all_tool_calls.append({"name": name, "arguments": args})

                # Execute the tool
                executor = self.tool_executors.get(name)
                if executor:
                    if name == "read_file":
                        result = executor(args.get("path"), self.base_path)
                    elif name == "write_file":
                        result = executor(args.get("path"), args.get("content"), self.base_path)
                    else:
                        result = executor(**args)
                else:
                    result = f"Error: Unknown tool '{name}'"

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result
                })

        # Max iterations reached
        return {
            "response": "Max tool iterations reached",
            "messages": messages,
            "tool_calls": all_tool_calls
        }
