"""Tool-calling agent for Ollama models with built-in file operations.

This module provides a ToolAgent class that enables LLM models to execute
tools (function calls) in an agentic loop. It includes built-in file
read/write tools with content-type validation and path security features.

Key Components:
    - ToolAgent: Main agent class that orchestrates the chat-tool loop
    - FILE_TOOLS: Built-in tool definitions for file operations
    - TOOL_EXECUTORS: Default executor functions for built-in tools
    - Content type detection and validation for file writes

Example:
    >>> from api_client import OllamaAPIClient
    >>> client = OllamaAPIClient()
    >>> agent = ToolAgent(client, model="qwen2.5-coder:7b")
    >>> result = agent.run("Read the contents of config.json")
    >>> print(result["response"])
"""

import json
import re
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


# Content type constants
CONTENT_TYPE_PYTHON = "python"
CONTENT_TYPE_JSON = "json"
CONTENT_TYPE_CSV = "csv"
CONTENT_TYPE_YAML = "yaml"
CONTENT_TYPE_SHELL = "shell"
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_MARKDOWN = "markdown"
CONTENT_TYPE_TEXT = "text"

# Mapping of content types to valid file extensions
CONTENT_TYPE_EXTENSIONS = {
    CONTENT_TYPE_PYTHON: {'.py'},
    CONTENT_TYPE_JSON: {'.json'},
    CONTENT_TYPE_CSV: {'.csv'},
    CONTENT_TYPE_YAML: {'.yaml', '.yml'},
    CONTENT_TYPE_SHELL: {'.sh', '.bash'},
    CONTENT_TYPE_HTML: {'.html', '.htm'},
    CONTENT_TYPE_MARKDOWN: {'.md', '.markdown'},
    CONTENT_TYPE_TEXT: {'.txt', '.text', ''},  # Allow no extension for plain text
}


def _looks_like_csv(content: str) -> bool:
    """Check if content appears to be CSV-formatted data.

    Uses heuristics based on comma consistency across lines to determine
    if content is likely CSV data rather than plain text.

    Args:
        content: The string content to analyze.

    Returns:
        True if the content appears to be CSV data, False otherwise.

    Examples:
        >>> _looks_like_csv("name,age,city\\nAlice,30,NYC\\nBob,25,LA")
        True

        >>> _looks_like_csv("This is just plain text.\\nNo commas here.")
        False
    """
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return False

    # Count commas per line (check first 10 lines)
    comma_counts = [line.count(',') for line in lines[:10]]
    if not comma_counts or comma_counts[0] == 0:
        return False

    # All lines should have similar comma count (allowing some variance)
    first_count = comma_counts[0]
    return all(abs(c - first_count) <= 1 for c in comma_counts)


def _detect_content_type(content: str) -> str:
    """Detect the type of content based on patterns and structure.

    Analyzes the content using a priority-ordered set of heuristics:
    1. JSON (via parsing attempt)
    2. Python (function/class definitions, imports, decorators)
    3. Shell scripts (shebang detection)
    4. CSV (comma-delimited structure)
    5. HTML (doctype and common tags)
    6. YAML (key-value patterns)
    7. Markdown (headers, links, lists, code blocks)
    8. Plain text (default fallback)

    Args:
        content: The string content to analyze.

    Returns:
        One of the CONTENT_TYPE_* module constants indicating the detected type.

    Examples:
        >>> _detect_content_type('{"name": "test", "value": 42}')
        'json'

        >>> _detect_content_type('def hello():\\n    print("Hello, World!")')
        'python'
    """
    content_stripped = content.strip()

    # Empty or whitespace-only content defaults to text
    if not content_stripped:
        return CONTENT_TYPE_TEXT

    # 1. JSON detection (try parsing first for structured data)
    if content_stripped.startswith('{') or content_stripped.startswith('['):
        try:
            parsed = json.loads(content_stripped)
            if isinstance(parsed, (dict, list)):
                return CONTENT_TYPE_JSON
        except json.JSONDecodeError:
            pass

    # 2. Python detection (high confidence indicators)
    python_patterns = [
        r'^\s*def\s+\w+\s*\(',           # function definition
        r'^\s*class\s+\w+.*:',           # class definition
        r'^\s*import\s+\w+',             # import statement
        r'^\s*from\s+\w+.*\s+import',    # from import
        r'if\s+__name__\s*==\s*["\']__main__["\']:', # main guard
        r'^\s*@\w+',                     # decorators
    ]
    for pattern in python_patterns:
        if re.search(pattern, content, re.MULTILINE):
            return CONTENT_TYPE_PYTHON

    # 3. Shell script detection
    first_line = content_stripped.split('\n')[0]
    if first_line.startswith('#!') and ('bash' in first_line or '/sh' in first_line):
        return CONTENT_TYPE_SHELL

    # 4. CSV detection
    if _looks_like_csv(content_stripped):
        return CONTENT_TYPE_CSV

    # 5. HTML detection
    html_patterns = [
        r'<!DOCTYPE\s+html',
        r'<html[\s>]',
        r'<head[\s>]',
        r'<body[\s>]',
    ]
    if any(re.search(p, content, re.IGNORECASE) for p in html_patterns):
        return CONTENT_TYPE_HTML

    # 6. YAML detection (after JSON check, look for key: value patterns)
    if not content_stripped.startswith(('{', '[')):
        key_value_lines = len(re.findall(r'^\s*[\w-]+:\s*', content, re.MULTILINE))
        if key_value_lines >= 2:
            return CONTENT_TYPE_YAML

    # 7. Markdown detection
    markdown_patterns = [
        r'^#{1,6}\s+\S',      # headers
        r'\[.+\]\(.+\)',      # links
        r'^\s*[-*]\s+\S',     # bullet points
        r'```',               # code blocks
    ]
    md_score = sum(1 for p in markdown_patterns if re.search(p, content, re.MULTILINE))
    if md_score >= 2:
        return CONTENT_TYPE_MARKDOWN

    # 8. Default to text
    return CONTENT_TYPE_TEXT


def _extract_code_blocks(content: str, language: str) -> str:
    """Extract fenced code blocks of a specified language from markdown content.

    Finds all code blocks marked with the specified language (or 'py' for Python)
    and concatenates them into a single string.

    Args:
        content: Markdown content potentially containing fenced code blocks.
        language: The language identifier to match (e.g., 'python', 'javascript').
                  For Python, also matches 'py' as an alias.

    Returns:
        Concatenated code from all matching code blocks, separated by double
        newlines. Returns empty string if no matching blocks found.

    Examples:
        >>> md = "# Example\\n```python\\nprint('hello')\\n```"
        >>> _extract_code_blocks(md, 'python')
        "print('hello')"

        >>> md = "No code blocks here, just text."
        >>> _extract_code_blocks(md, 'python')
        ''
    """
    # Match ```python or ```py code blocks
    pattern = rf'```(?:{language}|py)\s*\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    return '\n\n'.join(matches).strip()


def _validate_file_extension(path: str, content: str) -> tuple[Optional[str], Optional[str]]:
    """Validate that a file's extension matches its detected content type.

    This validation prevents common LLM mistakes like saving Python code with
    a .md extension or JSON data without any extension. It uses content-type
    detection to determine what the file should be saved as.

    Args:
        path: The file path provided by the model, including the desired extension.
        content: The content to be written to the file.

    Returns:
        A tuple of (error_message, suggested_path):
        - If validation passes: (None, None)
        - If validation fails: (error_message explaining the issue,
          suggested_path with the corrected extension)

    Examples:
        >>> _validate_file_extension("script.py", "def foo(): pass")
        (None, None)  # Valid: Python code with .py extension

        >>> _validate_file_extension("script.md", "def foo(): pass")
        ("Error: File 'script.md' has .md extension...", "script.py")
    """
    detected_type = _detect_content_type(content)
    file_path = Path(path)
    extension = file_path.suffix.lower()

    valid_extensions = CONTENT_TYPE_EXTENSIONS.get(detected_type, set())

    # Case 1: No extension provided
    if not extension:
        if detected_type != CONTENT_TYPE_TEXT:
            # Need an extension for non-text content
            suggested_ext = list(valid_extensions)[0]
            suggested_path = str(file_path) + suggested_ext
            return (
                f"Error: File '{path}' has no extension but content appears to be {detected_type}. "
                f"Please use extension '{suggested_ext}' (e.g., '{suggested_path}').",
                suggested_path
            )
        return (None, None)  # Plain text with no extension is fine

    # Case 2: Extension doesn't match content type
    if extension not in valid_extensions:
        # Special case: .md file with Python content - clear rejection
        if extension in {'.md', '.markdown'} and detected_type == CONTENT_TYPE_PYTHON:
            suggested_path = str(file_path.with_suffix('.py'))
            return (
                f"Error: File '{path}' has .md extension but content is Python code. "
                f"Python code must be saved with .py extension. "
                f"Please retry with path '{suggested_path}'.",
                suggested_path
            )

        # General mismatch case
        expected_ext = list(valid_extensions)[0] if valid_extensions else '.txt'
        suggested_path = str(file_path.with_suffix(expected_ext))
        return (
            f"Error: File extension '{extension}' does not match detected content type '{detected_type}'. "
            f"Expected one of: {', '.join(sorted(valid_extensions))}. "
            f"Please retry with the correct extension (e.g., '{suggested_path}').",
            suggested_path
        )

    # Case 3: Extension matches - validation passes
    return (None, None)


def _resolve_path(path: str, base_path: Optional[Path], restrict: bool) -> tuple[Path, Optional[str]]:
    """Resolve a file path and optionally enforce directory restrictions.

    Handles both absolute and relative paths, resolving relative paths against
    the provided base_path. When restrict=True, ensures the resolved path
    remains within the base directory (prevents directory traversal attacks).

    Args:
        path: The file path to resolve (absolute or relative).
        base_path: Base directory for resolving relative paths. If None,
                   relative paths are resolved against the current working directory.
        restrict: If True, validates that the resolved path is within base_path.
                  Returns an error if the path escapes the base directory.

    Returns:
        A tuple of (resolved_path, error_message):
        - If valid: (fully resolved Path object, None)
        - If restricted and path escapes base: (resolved Path, error message)

    Examples:
        >>> _resolve_path("data/file.txt", Path("/home/user"), restrict=False)
        (PosixPath('/home/user/data/file.txt'), None)

        >>> _resolve_path("../etc/passwd", Path("/home/user"), restrict=True)
        (PosixPath('/etc/passwd'), "Error: Access denied - path '../etc/passwd' is outside allowed directory")
    """
    file_path = Path(path)
    if base_path and not file_path.is_absolute():
        file_path = base_path / file_path

    file_path = file_path.resolve()

    if restrict and base_path:
        base_resolved = base_path.resolve()
        try:
            file_path.relative_to(base_resolved)
        except ValueError:
            return file_path, f"Error: Access denied - path '{path}' is outside allowed directory"

    return file_path, None


def _execute_read_file(path: str, base_path: Optional[Path] = None, restrict: bool = False) -> str:
    """Execute the read_file tool to retrieve file contents.

    Args:
        path: Path to the file to read.
        base_path: Base directory for resolving relative paths.
        restrict: If True, only allow reading files within base_path.

    Returns:
        The file contents as a string, or an error message if the file
        cannot be read (not found, permission denied, outside allowed directory).

    Examples:
        >>> _execute_read_file("config.json", Path("/app"))
        '{"debug": true, "port": 8080}'

        >>> _execute_read_file("missing.txt", Path("/app"))
        "Error: File '/app/missing.txt' not found"
    """
    file_path, error = _resolve_path(path, base_path, restrict)
    if error:
        return error

    if not file_path.exists():
        return f"Error: File '{file_path}' not found"

    try:
        return file_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file_internal(path: str, content: str, base_path: Optional[Path], restrict: bool) -> str:
    """Write content to a file without content-type validation.

    This is the low-level write function that handles path resolution,
    directory creation, and actual file writing. It does not perform
    content-type validation (use _execute_write_file for that).

    Args:
        path: Path to the file to write.
        content: String content to write to the file.
        base_path: Base directory for resolving relative paths.
        restrict: If True, only allow writing files within base_path.

    Returns:
        Success message with the resolved path, or an error message if
        the write fails (permission denied, outside allowed directory).

    Examples:
        >>> _write_file_internal("output.txt", "Hello!", Path("/tmp"), False)
        "Successfully wrote to '/tmp/output.txt'"

        >>> _write_file_internal("../etc/hosts", "hack", Path("/tmp"), True)
        "Error: Access denied - path '../etc/hosts' is outside allowed directory"
    """
    file_path, error = _resolve_path(path, base_path, restrict)
    if error:
        return error

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing file: {e}"


def _execute_write_file(path: str, content: str, base_path: Optional[Path] = None, restrict: bool = False) -> str:
    """Execute the write_file tool with content-type validation.

    Validates that the file extension matches the detected content type before
    writing. Special handling for markdown files containing Python code blocks:
    both the markdown and extracted Python code are saved.

    Args:
        path: Path to the file to write.
        content: String content to write to the file.
        base_path: Base directory for resolving relative paths.
        restrict: If True, only allow writing files within base_path.

    Returns:
        Success message(s) indicating files written, or an error message if
        validation fails or the write cannot be completed.

    Examples:
        >>> _execute_write_file("script.py", "def main(): pass", Path("/app"))
        "Successfully wrote to '/app/script.py'"

        >>> _execute_write_file("script.md", "def main(): pass", Path("/app"))
        "Error: File 'script.md' has .md extension but content is Python code..."
    """
    extension = Path(path).suffix.lower()

    # Special case: markdown files with Python code blocks
    # Always save both: the markdown file AND extracted Python code
    if extension in {'.md', '.markdown'}:
        code = _extract_code_blocks(content, 'python')
        if code:
            # Save both files
            md_result = _write_file_internal(path, content, base_path, restrict)
            py_path = str(Path(path).with_suffix('.py'))
            py_result = _write_file_internal(py_path, code, base_path, restrict)
            return f"{md_result}\n{py_result}"
        # No Python code blocks, just save the markdown
        return _write_file_internal(path, content, base_path, restrict)

    # Standard validation: content type must match extension
    error, _ = _validate_file_extension(path, content)
    if error:
        return error

    return _write_file_internal(path, content, base_path, restrict)
    

TOOL_EXECUTORS: Dict[str, Callable[..., str]] = {
    "read_file": _execute_read_file,
    "write_file": _execute_write_file
}


class ToolAgent:
    """Agent that orchestrates a chat loop with automatic tool execution.

    The ToolAgent wraps an Ollama chat API, enabling models to call tools
    (functions) and receive their results in an agentic loop. It continues
    the conversation until the model produces a final response without
    tool calls, or until max_iterations is reached.

    Built-in file tools (read_file, write_file) are provided by default,
    with configurable base paths and optional directory restrictions for
    security.

    Attributes:
        client: The OllamaAPIClient instance for API communication.
        model: The model identifier to use for chat completions.
        tools: List of tool definitions in OpenAI function-calling format.
        tool_executors: Dict mapping tool names to their executor functions.
        read_base_path: Base directory for resolving read_file paths.
        write_base_path: Base directory for resolving write_file paths.
        max_iterations: Maximum number of tool-calling iterations before stopping.
        timeout: Request timeout in seconds for API calls.
        restrict_to_base: If True, file operations are restricted to base paths.

    Example:
        >>> client = OllamaAPIClient()
        >>> agent = ToolAgent(
        ...     client=client,
        ...     model="qwen2.5-coder:7b",
        ...     write_base_path=Path("/tmp/sandbox"),
        ...     restrict_to_base=True
        ... )
        >>> result = agent.run("Create a hello.py file that prints 'Hello, World!'")
        >>> print(result["response"])
    """

    def __init__(self, client: OllamaAPIClient, model: str,
                 tools: Optional[List[Dict]] = None,
                 tool_executors: Optional[Dict[str, Callable]] = None,
                 read_base_path: Optional[Path] = None,
                 write_base_path: Optional[Path] = None,
                 max_iterations: int = 10,
                 timeout: Optional[float] = 600,
                 restrict_to_base: bool = False):
        """Initialize the ToolAgent.

        Args:
            client: OllamaAPIClient instance for making API requests.
            model: Model identifier (e.g., "qwen2.5-coder:7b", "llama3.1:8b").
            tools: Optional list of tool definitions. Defaults to FILE_TOOLS.
            tool_executors: Optional dict of tool name -> executor function.
                            Defaults to TOOL_EXECUTORS (read_file, write_file).
            read_base_path: Base path for read operations. Defaults to cwd.
            write_base_path: Base path for write operations. Defaults to cwd.
            max_iterations: Maximum tool-calling rounds before stopping. Default 10.
            timeout: API request timeout in seconds. Default 600 (10 minutes).
            restrict_to_base: If True, file ops must stay within base paths.
        """
        self.client = client
        self.model = model
        self.tools = tools or FILE_TOOLS
        self.tool_executors = tool_executors or TOOL_EXECUTORS
        self.read_base_path = read_base_path or Path.cwd()
        self.write_base_path = write_base_path or Path.cwd()
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.restrict_to_base = restrict_to_base

    def run(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent with a prompt, executing tools until completion.

        Initiates a chat loop where the model can make tool calls that are
        automatically executed. The loop continues until the model produces
        a response without tool calls, or max_iterations is reached.

        Args:
            prompt: The user's prompt or instruction for the agent.
            system: Optional system message to set context/behavior.

        Returns:
            A dict containing:
                - 'response': The model's final text response.
                - 'messages': Complete conversation history including tool results.
                - 'tool_calls': List of all tool calls made during execution,
                  each containing 'name' and 'arguments' keys.

        Examples:
            >>> agent = ToolAgent(client, "qwen2.5-coder:7b")
            >>> result = agent.run("What's in the README.md file?")
            >>> print(result["response"])
            "The README contains installation instructions and usage examples..."
            >>> print(result["tool_calls"])
            [{'name': 'read_file', 'arguments': {'path': 'README.md'}}]

            >>> result = agent.run(
            ...     "Create a Python script that prints the current date",
            ...     system="You are a helpful coding assistant."
            ... )
            >>> print(result["tool_calls"])
            [{'name': 'write_file', 'arguments': {'path': 'date_script.py', 'content': '...'}}]
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
                tools=self.tools,
                timeout=self.timeout
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
                        result = executor(args.get("path"), self.read_base_path, self.restrict_to_base)
                    elif name == "write_file":
                        result = executor(args.get("path"), args.get("content"), self.write_base_path, self.restrict_to_base)
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
