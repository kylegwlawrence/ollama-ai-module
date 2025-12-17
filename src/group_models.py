import subprocess
from typing import List, Dict

def get_installed_models() -> List[str]:
    """Get the list of locally installed Ollama models.

    Returns:
        A list of model names installed locally.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    try:
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        models = []

        # Skip the header line and parse model names
        for line in lines[1:]:
            if line.strip():
                # The first column is the model name
                model_name = line.split()[0]
                models.append(model_name)

        return models

    except FileNotFoundError:
        raise FileNotFoundError("Ollama is not installed or not in PATH")
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode,
            e.cmd,
            stderr=f"Error running 'ollama list': {e.stderr}"
        )

def get_models_grouped_by_base_name() -> Dict[str, List[str]]:
    """Get locally installed Ollama models grouped by their base name.

    Models are grouped by removing the tag (the part after the colon).
    For example:
        - llama3.2:1b and llama3.2:7b are grouped as llama3.2
        - qwen2.5:3b is grouped as qwen2.5

    Returns:
        A dictionary where keys are base model names and values are lists
        of full model names (with tags) for that base model.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    models = get_installed_models()
    grouped = {}

    for model in models:
        # Extract base name (part before the colon)
        base_name = model.split(':')[0]

        if base_name not in grouped:
            grouped[base_name] = []

        grouped[base_name].append(model)

    return grouped


def get_models_grouped_by_size() -> Dict[float, List[str]]:
    """Get locally installed Ollama models grouped by their parameter size.

    Models are grouped by extracting the size tag (the part after the colon)
    and converting it to billions. For example:
        - llama3.2:7b and qwen2.5:7b are grouped as 7.0 (7 billion)
        - smollm:135m is grouped as 0.135 (135 million)
        - llama3.2:1b is grouped as 1.0 (1 billion)

    Returns:
        A dictionary where keys are numeric sizes in billions and values are lists
        of full model names (with tags) for that size.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    models = get_installed_models()
    grouped = {}

    for model in models:
        # Extract size tag (part after the colon)
        parts = model.split(':')
        size_str = parts[1] if len(parts) > 1 else 'unknown'

        # Convert size to billions
        if size_str != 'unknown':
            size_value = float(size_str[:-1])  # Remove the letter and convert to float
            unit = size_str[-1].lower()  # Get the last character (b or m)

            if unit == 'b':
                # Already in billions
                size_numeric = size_value
            elif unit == 'm':
                # Convert millions to billions
                size_numeric = size_value / 1000
            else:
                # Unknown unit, skip
                continue
        else:
            size_numeric = 0.0

        if size_numeric not in grouped:
            grouped[size_numeric] = []

        grouped[size_numeric].append(model)

    return grouped


def get_models_in_size_range(min_size: float, max_size: float) -> List[str]:
    """Get models within a specific parameter size range (in billions).

    Args:
        min_size: Minimum parameter size in billions (e.g., 0.1 for 100M)
        max_size: Maximum parameter size in billions (e.g., 7.0 for 7B)

    Returns:
        A list of model names that fall within the specified size range.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
        ValueError: If min_size is greater than max_size.
    """
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot be greater than max_size ({max_size})")

    models = get_installed_models()
    filtered_models = []

    for model in models:
        # Extract size tag (part after the colon)
        parts = model.split(':')
        size_str = parts[1] if len(parts) > 1 else None

        # Convert size to billions
        if size_str:
            try:
                size_value = float(size_str[:-1])  # Remove the letter and convert to float
                unit = size_str[-1].lower()  # Get the last character (b or m)

                if unit == 'b':
                    size_numeric = size_value
                elif unit == 'm':
                    size_numeric = size_value / 1000
                else:
                    continue

                # Check if within range
                if min_size <= size_numeric <= max_size:
                    filtered_models.append(model)
            except (ValueError, IndexError):
                # Skip if unable to parse size
                continue

    return filtered_models
