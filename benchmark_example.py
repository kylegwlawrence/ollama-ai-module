from benchmark_models import benchmark_models

if __name__ == "__main__":
    benchmark_models(
        use_installed=True,
        prompts = [
                "write a haiku about the wind",
                "what is 2 factorial 2?",
                "What is the capital of the 5th most populous country in the world?",
                "write a simple python function to reverse a list then remove the second element, and return the new list"
            ]
        )