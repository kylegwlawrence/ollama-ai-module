import csv
import os
import time
import threading
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Any
from src.run_ollama import run_ollama_smart, ensure_ollama_server_running, get_ollama_server_metrics
from src.model import get_installed_models, check_and_install_model

def run_ollama_benchmark(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    Runs an Ollama model with a prompt and measures response time and resource usage.

    Args:
        model_name: Name of the Ollama model to use
        prompt: The prompt to send to the model

    Returns:
        Dictionary containing model_name, duration, response, and resource metrics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_samples = []
    memory_samples = []

    def monitor_resources() -> None:
        """Background thread to collect resource metrics."""
        while monitoring:
            metrics = get_ollama_server_metrics()
            if metrics:
                cpu_samples.append(metrics['cpu_percent'])
                memory_samples.append(metrics['memory_mb'])
            time.sleep(0.1)  # Sample every 100ms

    try:
        # Start monitoring
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()

        start_time = time.time()

        # Use smart runner to leverage already-running models via API
        response = run_ollama_smart(model_name, prompt, return_output=True)

        end_time = time.time()
        duration = end_time - start_time

        # Stop monitoring
        monitoring = False
        monitor_thread.join(timeout=1)

        # Calculate statistics
        if cpu_samples and memory_samples:
            cpu_avg = round(sum(cpu_samples) / len(cpu_samples), 2)
            cpu_peak = round(max(cpu_samples), 2)
            memory_avg = round(sum(memory_samples) / len(memory_samples), 2)
            memory_peak = round(max(memory_samples), 2)
        else:
            cpu_avg = cpu_peak = memory_avg = memory_peak = None

        return {
            'model_name': model_name,
            'prompt': prompt,
            'duration_seconds': round(duration, 2),
            'cpu_avg_percent': cpu_avg,
            'cpu_peak_percent': cpu_peak,
            'memory_avg_mb': memory_avg,
            'memory_peak_mb': memory_peak,
            'response': response.strip().replace('\n', ' '),
            'timestamp': timestamp
        }

    except Exception as e:
        # Stop monitoring on error
        monitoring = False
        end_time = time.time()
        duration = end_time - start_time if 'start_time' in locals() else 0

        return {
            'model_name': model_name,
            'prompt': prompt,
            'duration_seconds': round(duration, 2),
            'cpu_avg_percent': None,
            'cpu_peak_percent': None,
            'memory_avg_mb': None,
            'memory_peak_mb': None,
            'response': f"ERROR: {e}".replace('\n', ' '),
            'timestamp': timestamp
        }

def benchmark_models(models: Optional[List[str]] = None, prompts: Optional[List[str]] = None, output_file: Optional[str] = None, use_installed: bool = False) -> None:
    """
    Benchmarks multiple models with multiple prompts and saves results to CSV.

    Args:
        models: List of model names to benchmark (optional if use_installed=True)
        prompts: List of prompts to test all models with
        output_file: Path to the output CSV file (default: generates timestamped filename)
        use_installed: If True, uses all installed models instead of custom list
    """
    # Ensure Ollama server is running before benchmarking
    ensure_ollama_server_running()

    # Generate timestamped filename if not provided
    if output_file is None:
        benchmark_dir = "benchmarks"
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(benchmark_dir, f"model_benchmark_{timestamp}.csv")
    # Determine which models to use
    if use_installed:
        models_to_benchmark = get_installed_models()
        if not models_to_benchmark:
            raise ValueError("No installed models found.")
    elif models and len(models) > 0:
        models_to_benchmark = models
        # Check and install each model if needed
        for model in models_to_benchmark:
            check_and_install_model(model)
    else:
        raise ValueError("Either provide a list of models or set use_installed=True")

    # Run benchmarking
    results = []
    for model in models_to_benchmark:
        for prompt in prompts:
            result = run_ollama_benchmark(model, prompt)
            results.append(result)

        # Stop the model to free resources after all prompts
        try:
            subprocess.run(['ollama', 'stop', model], check=False, capture_output=True)
            print(f"Model '{model}' stopped successfully")
        except Exception:
            pass  # Ignore errors if model is already stopped

    # Write results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'model_name',
            'prompt',
            'duration_seconds',
            'cpu_avg_percent',
            'cpu_peak_percent',
            'memory_avg_mb',
            'memory_peak_mb',
            'response',
            'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to: {output_file}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark AI models with multiple prompts')
    parser.add_argument('prompts', type=str, nargs='+', help='One or more prompts to test all models with')
    parser.add_argument('-m', '--models', nargs='+', help='List of model names (if not provided, uses all installed models)')

    args = parser.parse_args()

    # Determine models to use
    if args.models:
        print(f"Using specified models: {', '.join(args.models)}")
        benchmark_models(models=args.models, prompts=args.prompts)
    else:
        print("Using all installed models")
        benchmark_models(prompts=args.prompts, use_installed=True)

if __name__ == "__main__":
    main()
