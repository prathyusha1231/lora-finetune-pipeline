#!/usr/bin/env python3
"""
Evaluate a trained LoRA model checkpoint.

This script loads a trained LoRA adapter and evaluates it on benchmark tasks,
computing metrics like perplexity, accuracy, and inference performance.

Usage examples:
    # Evaluate a trained checkpoint
    python scripts/evaluate_model.py \
        --checkpoint output/rank_ablation_r16/final_model \
        --base-model microsoft/phi-2 \
        --dataset data/alpaca_sample.jsonl

    # Evaluate with custom benchmark prompts
    python scripts/evaluate_model.py \
        --checkpoint output/rank_ablation_r16/final_model \
        --base-model microsoft/phi-2 \
        --benchmark-only

    # Profile inference performance
    python scripts/evaluate_model.py \
        --checkpoint output/rank_ablation_r16/final_model \
        --base-model microsoft/phi-2 \
        --profile-inference
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.profiler import MemoryProfiler, InferenceProfiler
from src.evaluation.benchmarks import BenchmarkSuite
from src.data.dataset import load_dataset_from_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LoRA model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint directory"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace model ID of the base model"
    )

    # Evaluation options
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to test dataset (JSONL format) for perplexity evaluation"
    )

    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run benchmark prompts (skip dataset evaluation)"
    )

    parser.add_argument(
        "--profile-inference",
        action="store_true",
        help="Profile inference speed and memory usage"
    )

    parser.add_argument(
        "--num-benchmark-samples",
        type=int,
        default=10,
        help="Number of benchmark prompts to test (default: 10)"
    )

    # Model loading options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to load model on (default: auto)"
    )

    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )

    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results JSON (default: print to console)"
    )

    return parser.parse_args()


def load_model_and_lora(args):
    """Load base model and LoRA adapter."""
    print(f"Loading base model: {args.base_model}")

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load base model with optional quantization
    load_kwargs = {"device_map": device}

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        **load_kwargs
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on: {device}")
    return model, tokenizer, device


def evaluate_on_dataset(model, tokenizer, dataset_path):
    """Evaluate model perplexity on a dataset."""
    print(f"\nEvaluating on dataset: {dataset_path}")

    # Load dataset
    dataset = load_dataset_from_file(dataset_path)
    print(f"Dataset size: {len(dataset)} examples")

    # Use last 20% as test set (or up to 1000 examples)
    test_size = min(len(dataset) // 5, 1000)
    test_dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))

    print(f"Test set size: {len(test_dataset)} examples")

    # Calculate metrics
    metrics_calculator = MetricsCalculator()
    metrics = metrics_calculator.compute_perplexity(model, tokenizer, test_dataset)

    return metrics


def run_benchmark_prompts(model, tokenizer, num_samples=10):
    """Run benchmark prompts and generate responses."""
    print(f"\nRunning {num_samples} benchmark prompts...")

    benchmark = BenchmarkSuite()
    prompts = benchmark.get_benchmark_prompts()[:num_samples]

    results = []

    for idx, prompt_data in enumerate(prompts, 1):
        prompt = prompt_data["prompt"]
        category = prompt_data["category"]

        print(f"\n[{idx}/{num_samples}] Category: {category}")
        print(f"Prompt: {prompt[:100]}...")

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        print(f"Response: {response[:200]}...")

        results.append({
            "category": category,
            "prompt": prompt,
            "response": response,
        })

    return results


def profile_inference(model, tokenizer, num_runs=50):
    """Profile inference speed and memory usage."""
    print(f"\nProfiling inference performance ({num_runs} runs)...")

    profiler = InferenceProfiler()

    # Test prompt
    test_prompt = "Explain the concept of machine learning in simple terms."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)

    # Profile
    latencies = []
    memory_usage = []

    memory_profiler = MemoryProfiler()

    for i in range(num_runs):
        memory_profiler.start()
        latency = profiler.measure_latency(
            model,
            inputs,
            max_new_tokens=50,
        )
        mem = memory_profiler.get_peak_memory()

        latencies.append(latency)
        memory_usage.append(mem)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs...")

    # Calculate statistics
    import statistics
    avg_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
    avg_memory = statistics.mean(memory_usage)

    # Estimate tokens/second (rough estimate: 50 tokens / latency)
    tokens_per_second = 50 / avg_latency if avg_latency > 0 else 0

    profiling_results = {
        "average_latency_ms": avg_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "tokens_per_second": tokens_per_second,
        "average_memory_gb": avg_memory,
        "num_runs": num_runs,
    }

    return profiling_results


def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if "metrics" in results:
        print("\nDataset Metrics:")
        for name, value in results["metrics"].items():
            print(f"  {name}: {value:.4f}")

    if "benchmark_results" in results:
        print(f"\nBenchmark Prompts: {len(results['benchmark_results'])} responses generated")
        print("(See full results in output JSON)")

    if "profiling" in results:
        print("\nInference Profiling:")
        for name, value in results["profiling"].items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Validate dataset if provided
    if args.dataset and not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Load model
    try:
        model, tokenizer, device = load_model_and_lora(args)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Collect results
    results = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "device": device,
    }

    # Dataset evaluation
    if args.dataset and not args.benchmark_only:
        try:
            metrics = evaluate_on_dataset(model, tokenizer, args.dataset)
            results["metrics"] = metrics
        except Exception as e:
            print(f"Error evaluating on dataset: {str(e)}")
            import traceback
            traceback.print_exc()

    # Benchmark prompts
    if args.benchmark_only or not args.dataset:
        try:
            benchmark_results = run_benchmark_prompts(
                model, tokenizer, num_samples=args.num_benchmark_samples
            )
            results["benchmark_results"] = benchmark_results
        except Exception as e:
            print(f"Error running benchmarks: {str(e)}")
            import traceback
            traceback.print_exc()

    # Inference profiling
    if args.profile_inference:
        try:
            profiling_results = profile_inference(model, tokenizer)
            results["profiling"] = profiling_results
        except Exception as e:
            print(f"Error profiling inference: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
