import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from src.run_ollama import run_ollama
from src.select_model import get_installed_models
from src.install_ollama_model import check_ollama_model_installed


def run_ollama_benchmark(model_name: str, prompt: str) -> Dict[str, any]:
    """
    Runs an Ollama model with a prompt and measures response time.

    Args:
        model_name: Name of the Ollama model to use
        prompt: The prompt to send to the model

    Returns:
        Dictionary containing model_name, duration, response, and timestamp
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        response = run_ollama(model_name, prompt, return_output=True)
        end_time = time.time()
        duration = end_time - start_time

        return {
            'model_name': model_name,
            'duration_seconds': round(duration, 2),
            'response': response.strip(),
            'timestamp': timestamp
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        return {
            'model_name': model_name,
            'duration_seconds': round(duration, 2),
            'response': f"ERROR: {e}",
            'timestamp': timestamp
        }


def benchmark_models(models: Optional[List[str]] = None, prompt: str = None, output_file: str = "model_benchmark.csv", use_installed: bool = False):
    """
    Benchmarks multiple models with the same prompt and saves results to CSV.

    Args:
        models: List of model names to benchmark (optional if use_installed=True)
        prompt: The prompt to test all models with
        output_file: Path to the output CSV file (default: model_benchmark.csv)
        use_installed: If True, uses all installed models instead of custom list
    """
    # Determine which models to use
    if use_installed:
        models_to_benchmark = get_installed_models()
        if not models_to_benchmark:
            raise ValueError("No installed models found.")
    elif models and len(models) > 0:
        models_to_benchmark = models
        # Check and install each model if needed
        for model in models_to_benchmark:
            check_ollama_model_installed(model)
    else:
        raise ValueError("Either provide a list of models or set use_installed=True")

    # Run benchmarking
    results = []
    for model in models_to_benchmark:
        result = run_ollama_benchmark(model, prompt)
        results.append(result)

    # Write results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'duration_seconds', 'response', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark AI models with the same prompt')
    parser.add_argument('prompt', type=str, help='The prompt to test all models with')
    parser.add_argument('-m', '--models', nargs='+', help='List of model names (if not provided, uses all installed models)')
    parser.add_argument('-o', '--output', type=str, default='model_benchmark.csv', help='Output CSV filename (default: model_benchmark.csv)')

    args = parser.parse_args()

    # Determine models to use
    if args.models:
        print(f"Using specified models: {', '.join(args.models)}")
        benchmark_models(models=args.models, prompt=args.prompt, output_file=args.output)
    else:
        print("Using all installed models")
        benchmark_models(prompt=args.prompt, output_file=args.output, use_installed=True)


if __name__ == "__main__":
    main()
