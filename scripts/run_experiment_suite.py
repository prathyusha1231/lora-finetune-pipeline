#!/usr/bin/env python3
"""
Run a suite of LoRA experiments from a configuration file.

This script automates running multiple experiments with different configurations,
logging all results to a database for later analysis.

Usage examples:
    # Run rank ablation suite
    python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml

    # Run module ablation suite
    python scripts/run_experiment_suite.py --suite configs/experiments/module_sweep.yaml

    # Run quantization study
    python scripts/run_experiment_suite.py --suite configs/experiments/quantization_sweep.yaml

    # Run all experiments
    python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml --continue-on-error
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
import time

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
        description="Run a suite of LoRA experiments from a YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Path to experiment suite YAML configuration file"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="results/experiments.db",
        help="Path to experiments database (default: results/experiments.db)"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running experiments even if one fails"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually running experiments"
    )

    return parser.parse_args()


def load_suite_config(config_path: str) -> Dict[str, Any]:
    """Load experiment suite configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Suite configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["suite_name", "experiments"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in suite config: {field}")

    return config


def infer_experiment_type(suite_config: Dict[str, Any], experiment_config: Dict[str, Any]) -> str:
    """Infer experiment type from configuration."""
    # Check explicit type field
    if "type" in experiment_config:
        return experiment_config["type"]

    # Infer from suite name
    suite_name = suite_config.get("suite_name", "").lower()
    if "rank" in suite_name:
        return "rank_ablation"
    elif "module" in suite_name:
        return "module_ablation"
    elif "quantization" in suite_name or "quant" in suite_name:
        return "quantization"

    # Infer from parameters
    if "lora_rank" in experiment_config:
        return "rank_ablation"
    elif "target_modules" in experiment_config:
        return "module_ablation"
    elif "quantization_mode" in experiment_config:
        return "quantization"

    raise ValueError(f"Cannot infer experiment type from config: {experiment_config}")


def create_experiment_from_config(
    suite_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    experiment_idx: int
) -> tuple:
    """
    Create an experiment instance from configuration.

    Returns:
        Tuple of (ExperimentConfig, Experiment instance)
    """
    # Get common configuration
    base_model = experiment_config.get("base_model", suite_config.get("base_model"))
    dataset_path = experiment_config.get("dataset_path", suite_config.get("dataset_path"))
    output_dir = experiment_config.get("output_dir", suite_config.get("output_dir", "output"))

    # Training settings
    num_train_epochs = experiment_config.get(
        "num_train_epochs",
        suite_config.get("num_train_epochs", 3)
    )
    batch_size = experiment_config.get(
        "per_device_train_batch_size",
        suite_config.get("per_device_train_batch_size", 4)
    )
    max_seq_length = experiment_config.get(
        "max_seq_length",
        suite_config.get("max_seq_length", 512)
    )
    seed = experiment_config.get("seed", suite_config.get("seed", 42))

    # Infer experiment type
    experiment_type = infer_experiment_type(suite_config, experiment_config)

    # Create experiment name
    suite_name = suite_config["suite_name"]
    exp_name = experiment_config.get("name", f"exp_{experiment_idx}")

    # Create ExperimentConfig
    config = ExperimentConfig(
        experiment_name=f"{suite_name}_{exp_name}",
        experiment_type=experiment_type,
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        max_seq_length=max_seq_length,
        seed=seed,
    )

    # Create appropriate experiment instance
    if experiment_type == "rank_ablation":
        lora_rank = experiment_config.get("lora_rank")
        if lora_rank is None:
            raise ValueError(f"lora_rank required for rank_ablation experiment: {experiment_config}")
        experiment = RankAblationExperiment(config, lora_rank=lora_rank)

    elif experiment_type == "module_ablation":
        target_modules = experiment_config.get("target_modules")
        if target_modules is None:
            raise ValueError(f"target_modules required for module_ablation: {experiment_config}")
        experiment = ModuleAblationExperiment(config, target_modules=target_modules)

    elif experiment_type == "quantization":
        quantization_mode = experiment_config.get("quantization_mode")
        if quantization_mode is None:
            raise ValueError(f"quantization_mode required for quantization: {experiment_config}")
        experiment = QuantizationExperiment(config, quantization_mode=quantization_mode)

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    return config, experiment


def print_suite_info(suite_config: Dict[str, Any], experiments: List[tuple]):
    """Print suite configuration before running."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE CONFIGURATION")
    print("=" * 80)
    print(f"Suite Name: {suite_config['suite_name']}")
    print(f"Base Model: {suite_config.get('base_model', 'N/A')}")
    print(f"Dataset: {suite_config.get('dataset_path', 'N/A')}")
    print(f"Number of Experiments: {len(experiments)}")
    print(f"\nExperiments to run:")
    for idx, (config, exp) in enumerate(experiments, 1):
        print(f"  {idx}. {config.experiment_name} ({config.experiment_type})")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Load suite configuration
    try:
        suite_config = load_suite_config(args.suite)
    except Exception as e:
        print(f"Error loading suite configuration: {str(e)}")
        sys.exit(1)

    # Validate dataset exists
    dataset_path = suite_config.get("dataset_path")
    if dataset_path and not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Create experiments from configuration
    experiments = []
    for idx, exp_config in enumerate(suite_config["experiments"]):
        try:
            config, experiment = create_experiment_from_config(suite_config, exp_config, idx)
            experiments.append((config, experiment))
        except Exception as e:
            print(f"Error creating experiment {idx}: {str(e)}")
            if not args.continue_on_error:
                sys.exit(1)

    if not experiments:
        print("No experiments to run!")
        sys.exit(1)

    # Print suite info
    print_suite_info(suite_config, experiments)

    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No experiments will be executed\n")
        for idx, (config, exp) in enumerate(experiments, 1):
            print(f"Would run experiment {idx}/{len(experiments)}:")
            print(f"  Name: {config.experiment_name}")
            print(f"  Type: {config.experiment_type}")
            print(f"  Model: {config.base_model}")
            print(f"  Dataset: {config.dataset_path}")
            print()
        return

    # Initialize tracker
    tracker = ExperimentTracker(db_path=args.db_path)

    # Run experiments
    results = []
    successful = 0
    failed = 0

    suite_start_time = time.time()

    for idx, (config, experiment) in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT {idx}/{len(experiments)}")
        print(f"{'='*80}\n")

        try:
            # Run experiment
            result = experiment.run()

            # Save results to JSON
            results_path = os.path.join(
                config.output_dir,
                config.experiment_name,
                "results.json"
            )
            result.save(results_path)

            # Log to database
            exp_id = tracker.log_experiment(result.to_dict())
            print(f"\nExperiment logged to database with ID: {exp_id}")

            results.append((config.experiment_name, "SUCCESS", result))
            successful += 1

        except Exception as e:
            print(f"\nERROR in experiment {config.experiment_name}: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append((config.experiment_name, "FAILED", str(e)))
            failed += 1

            if not args.continue_on_error:
                print("\nAborting suite due to error (use --continue-on-error to continue)")
                break

    suite_end_time = time.time()
    total_time = suite_end_time - suite_start_time

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 80)
    print(f"Suite: {suite_config['suite_name']}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nDatabase: {args.db_path}")
    print("\nExperiment Results:")

    for name, status, result in results:
        if status == "SUCCESS":
            # Print metrics if available
            metrics_str = ""
            if hasattr(result, 'metrics') and result.metrics:
                metrics_str = " - " + ", ".join(
                    f"{k}: {v:.4f}" for k, v in list(result.metrics.items())[:2]
                )
            print(f"  ✓ {name}{metrics_str}")
        else:
            print(f"  ✗ {name} - {result}")

    print("=" * 80 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
