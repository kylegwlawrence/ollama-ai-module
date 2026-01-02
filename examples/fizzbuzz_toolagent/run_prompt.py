#!/usr/bin/env python3
"""
Simple script to process the prompt.md file using ToolAgent with file read/write capabilities.

Usage:
    python prompt_file.py [--model MODEL_NAME] [--timeout SECONDS]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.server import OllamaServer
from src.api_client import OllamaAPIClient
from src.tool_agent import ToolAgent


def main():
    parser = argparse.ArgumentParser(
        description="Process prompt.md using ToolAgent with file read/write tools."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen2.5-coder:1.5b-instruct-q5_K_M",
        help="Model name to use"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=6*60*60,
        help="Timeout in seconds"
    )

    args = parser.parse_args()

    # Read the prompt file from the same directory
    prompt_path = Path(__file__).parent / "prompt.md"
    if not prompt_path.exists():
        print(f"Error: File 'prompt.md' not found in {prompt_path.parent}", file=sys.stderr)
        sys.exit(1)

    prompt_content = prompt_path.read_text()

    if not prompt_content.strip():
        print("Error: prompt.md is empty.", file=sys.stderr)
        sys.exit(1)

    # Build the full prompt with instructions to write Python files
    prompt = f"""You have access to file read and write tools. Your task is to implement the code described in the specification below.

IMPORTANT: You must use the write_file tool to create each Python file. Write complete, working Python code to .py files in the current directory.

For example, if the spec calls for app.py, use write_file with path "app.py" and the full Python code as content.

Create all the necessary Python files as described in the specification. Do not just describe what you would write - actually write the files using the write_file tool.

Here is the specification to implement:

{prompt_content}

Now implement the code by writing each Python file using the write_file tool."""

    # Get model name
    if args.model:
        model_name = args.model
    else:
        server = OllamaServer()
        installed_models = server.get_installed_models()
        if not installed_models:
            print("Error: No models installed. Please install a model first.", file=sys.stderr)
            sys.exit(1)
        model_name = installed_models[0]

    # Run prompt with ToolAgent
    client = OllamaAPIClient()
    agent = ToolAgent(
        client,
        model=model_name,
        read_base_path=Path(__file__).parent.parent / "src",  # hello-ollama/src/ for reading modules
        write_base_path=Path(__file__).parent,  # recyclables/ for writing output files
        timeout=args.timeout
    )
    result = agent.run(prompt)

    response = result.get('response', '')
    tool_calls = result.get('tool_calls', [])

    # Write response to file
    response_path = Path(__file__).parent / "response.md"
    response_path.write_text(response)
    print(f"Response written to {response_path}")

    if tool_calls:
        print(f"Tools called: {[tc['name'] for tc in tool_calls]}")


if __name__ == "__main__":
    main()
