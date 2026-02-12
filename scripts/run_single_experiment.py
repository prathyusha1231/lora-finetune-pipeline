#!/usr/bin/env python3
"""
Run a single LoRA experiment with specified configuration.

Usage examples:
    # Rank ablation experiment
    python scripts/run_single_experiment.py \
        --experiment-type rank_ablation \
        --lora-rank 16 \
        --dataset data/alpaca_sample.jsonl \
        --base-model "microsoft/phi-2"

    # Module ablation experiment
    python scripts/run_single_experiment.py \
        --experiment-type module_ablation \
        --target-modules q_proj v_proj k_proj o_proj \
        --dataset data/alpaca_sample.jsonl \
        --base-model "microsoft/phi-2"

    # Quantization study
    python scripts/run_single_experiment.py \
        --experiment-type quantization \
        --quantization-mode 4bit \
        --dataset data/alpaca_sample.jsonl \
        --base-model "microsoft/phi-2"
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.base_experiment import ExperimentConfig
from src.experiments.rank_ablation import RankAblationExperiment
from src.experiments.module_ablation import ModuleAblationExperiment
from src.experiments.quantization_study import QuantizationExperiment
from src.utils.experiment_tracker import ExperimentTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a single LoRA efficiency experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--experiment-type",
        type=str,
        required=True,
        choices=["rank_ablation", "module_ablation", "quantization"],
        help="Type of experiment to run"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL format)"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model ID (default: microsoft/phi-2)"
    )

    # Experiment-specific arguments
    parser.add_argument(
        "--lora-rank",
        type=int,
        help="LoRA rank for rank_ablation experiment (e.g., 4, 8, 16, 32, 64)"
    )

    parser.add_argument(
        "--target-modules",
        nargs='+',
        help="Target modules for module_ablation (e.g., q_proj v_proj)"
    )

    parser.add_argument(
        "--quantization-mode",
        type=str,
        choices=["4bit", "8bit", "fp16"],
        help="Quantization mode for quantization experiment"
    )

    # Training configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for trained models (default: output)"
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4)"
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Database tracking
    parser.add_argument(
        "--db-path",
        type=str,
        default="results/experiments.db",
        help="Path to experiments database (default: results/experiments.db)"
    )

    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Don't save results to database"
    )

    return parser.parse_args()


def create_experiment_config(args) -> ExperimentConfig:
    """Create ExperimentConfig from command line arguments."""
    experiment_name = f"{args.experiment_type}_{args.base_model.split('/')[-1]}"

    if args.experiment_type == "rank_ablation":
        experiment_name += f"_r{args.lora_rank}"
    elif args.experiment_type == "module_ablation":
        modules_str = "_".join(args.target_modules)
        experiment_name += f"_{modules_str}"
    elif args.experiment_type == "quantization":
        experiment_name += f"_{args.quantization_mode}"

    return ExperimentConfig(
        experiment_name=experiment_name,
        experiment_type=args.experiment_type,
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )


def create_experiment(args, config: ExperimentConfig):
    """Create the appropriate experiment class based on type."""
    if args.experiment_type == "rank_ablation":
        if args.lora_rank is None:
            raise ValueError("--lora-rank is required for rank_ablation experiment")
        return RankAblationExperiment(config, lora_rank=args.lora_rank)

    elif args.experiment_type == "module_ablation":
        if args.target_modules is None:
            raise ValueError("--target-modules is required for module_ablation experiment")
        return ModuleAblationExperiment(config, target_modules=args.target_modules)

    elif args.experiment_type == "quantization":
        if args.quantization_mode is None:
            raise ValueError("--quantization-mode is required for quantization experiment")
        return QuantizationExperiment(config, quantization_mode=args.quantization_mode)

    else:
        raise ValueError(f"Unknown experiment type: {args.experiment_type}")


def print_experiment_info(args, config: ExperimentConfig):
    """Print experiment configuration before running."""
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Experiment Type: {config.experiment_type}")
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Base Model: {config.base_model}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output Directory: {config.output_dir}")
    print(f"\nTraining Settings:")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch Size: {config.per_device_train_batch_size}")
    print(f"  Max Sequence Length: {config.max_seq_length}")
    print(f"  Random Seed: {config.seed}")

    if args.experiment_type == "rank_ablation":
        print(f"\nExperiment Parameters:")
        print(f"  LoRA Rank: {args.lora_rank}")
    elif args.experiment_type == "module_ablation":
        print(f"\nExperiment Parameters:")
        print(f"  Target Modules: {', '.join(args.target_modules)}")
    elif args.experiment_type == "quantization":
        print(f"\nExperiment Parameters:")
        print(f"  Quantization Mode: {args.quantization_mode}")

    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Validate dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Create experiment configuration
    config = create_experiment_config(args)

    # Print configuration
    print_experiment_info(args, config)

    # Create and run experiment
    try:
        experiment = create_experiment(args, config)
        result = experiment.run()

        # Save results to JSON
        results_path = os.path.join(config.output_dir, config.experiment_name, "results.json")
        result.save(results_path)
        print(f"\nResults saved to: {results_path}")

        # Log to database (unless disabled)
        if not args.no_tracking:
            tracker = ExperimentTracker(db_path=args.db_path)
            exp_id = tracker.log_experiment(result.to_dict())
            print(f"Experiment logged to database with ID: {exp_id}")
            print(f"Database: {args.db_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Status: {result.metadata.get('status', 'unknown')}")
        print(f"\nMetrics:")
        for name, value in result.metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"\nProfiling:")
        for name, value in result.profiling.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
