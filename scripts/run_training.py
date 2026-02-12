#!/usr/bin/env python3
"""
Quick start script for LoRA fine-tuning.
Usage: python scripts/run_training.py --config config/default_config.yaml --dataset data.jsonl
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import LoRATrainingConfig, LoRATrainer


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/run_training.py --config config/default_config.yaml --dataset my_data.jsonl

  # With custom text column
  python scripts/run_training.py --config config/default_config.yaml --dataset data.csv --text_column content

  # Using HuggingFace dataset
  python scripts/run_training.py --config config/default_config.yaml --dataset tatsu-lab/alpaca
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (local file or HuggingFace dataset name)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in your dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set seed
    from src.utils import seed_everything
    seed_everything(args.seed)

    print("=" * 60)
    print("LoRA Fine-tuning Pipeline")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading config from: {args.config}")
    config = LoRATrainingConfig.from_yaml(args.config)

    # Print config summary
    print(f"\nConfiguration:")
    print(f"  Base Model: {config.base_model}")
    print(f"  LoRA Rank: {config.lora_r}")
    print(f"  LoRA Alpha: {config.lora_alpha}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch Size: {config.per_device_train_batch_size}")
    print(f"  Grad Accum Steps: {config.gradient_accumulation_steps}")
    print(f"  4-bit Quantization: {config.use_4bit}")

    # Initialize trainer
    trainer = LoRATrainer(config)

    # Load model
    print("\n" + "-" * 60)
    trainer.load_model()

    # Prepare dataset
    print("\n" + "-" * 60)
    dataset = trainer.prepare_dataset(args.dataset, args.text_column)
    print(f"Dataset size: {len(dataset)} samples")

    # Train
    print("\n" + "-" * 60)
    trainer.train(dataset)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
