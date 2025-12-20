import human_id


def generate_human_id(num_words: int = 3) -> str:
    """Generate a human-readable ID using random words.

    Args:
        num_words: Number of words to use in the ID (default: 3).

    Returns:
        A string containing a human-readable ID (e.g., "happy-blue-penguin").

    Raises:
        ValueError: If num_words is less than 3.
    """

    if num_words < 3:
        raise ValueError(f"num_words must be at least 3, got {num_words}")

    return human_id.generate_id(word_count=num_words)