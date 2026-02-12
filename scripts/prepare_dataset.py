#!/usr/bin/env python3
"""
Prepare datasets for LoRA fine-tuning experiments.

This script downloads and formats popular instruction-following datasets from HuggingFace,
creating JSONL files ready for training.

Supported datasets:
- alpaca: Stanford's instruction-following dataset (52K examples)
- alpaca-cleaned: Community-cleaned version of Alpaca
- dolly: Databricks instruction dataset (15K examples)
- oasst1: OpenAssistant Conversations dataset (high quality, conversational)
- wizard-evol: WizardLM Evol-Instruct V2 (196K examples, complex instructions)
- sharegpt: ShareGPT conversations (real ChatGPT conversations)

Usage examples:
    # Download Alpaca dataset (1000 sample)
    python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000

    # Download full Alpaca dataset
    python scripts/prepare_dataset.py --dataset alpaca --num-samples -1

    # Download OpenAssistant dataset (high quality)
    python scripts/prepare_dataset.py --dataset oasst1 --num-samples 1000

    # Download WizardLM (complex instructions)
    python scripts/prepare_dataset.py --dataset wizard-evol --num-samples 1000

    # With train/test split
    python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000 --train-test-split 0.9

    # Custom output path
    python scripts/prepare_dataset.py --dataset alpaca --output data/my_dataset.jsonl
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for LoRA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["alpaca", "alpaca-cleaned", "dolly", "oasst1", "wizard-evol", "sharegpt", "custom"],
        help="Dataset to download and prepare"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use (-1 for all, default: 1000)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file path (default: data/<dataset>_sample.jsonl)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="alpaca",
        choices=["alpaca", "chatml", "raw"],
        help="Output format (default: alpaca)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )

    parser.add_argument(
        "--train-test-split",
        type=float,
        help="Create train/test split (e.g., 0.9 for 90%% train, 10%% test)"
    )

    return parser.parse_args()


def format_alpaca_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    """
    Format example in Alpaca instruction format.

    Template:
        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
    """
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

    return prompt


def download_alpaca(cleaned: bool = False) -> List[Dict[str, Any]]:
    """Download Alpaca dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Install with: pip install datasets")
        sys.exit(1)

    print(f"Downloading {'cleaned ' if cleaned else ''}Alpaca dataset...")

    if cleaned:
        # Use community-cleaned version
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    else:
        # Use original version
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    print(f"Downloaded {len(dataset)} examples")

    # Convert to list of dicts
    examples = []
    for item in dataset:
        examples.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
        })

    return examples


def download_dolly() -> List[Dict[str, Any]]:
    """Download Dolly dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Install with: pip install datasets")
        sys.exit(1)

    print("Downloading Dolly dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    print(f"Downloaded {len(dataset)} examples")

    # Convert to Alpaca format
    examples = []
    for item in dataset:
        examples.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("context", ""),  # Dolly uses "context" instead of "input"
            "output": item.get("response", ""),
        })

    return examples


def download_oasst1() -> List[Dict[str, Any]]:
    """Download OpenAssistant Conversations dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Install with: pip install datasets")
        sys.exit(1)

    print("Downloading OpenAssistant (OASST1) dataset...")
    # Load the English subset
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    print(f"Downloaded {len(dataset)} examples")
    print("Processing conversation threads...")

    # OASST1 is a tree of conversations, extract user-assistant pairs
    examples = []
    for item in dataset:
        if item.get("role") == "assistant" and item.get("lang") == "en":
            # Get parent message (user prompt)
            parent_id = item.get("parent_id")

            # For simplicity, use the assistant response as output
            # In a more sophisticated version, we'd reconstruct full conversations
            examples.append({
                "instruction": item.get("text", "")[:500],  # Truncate very long responses
                "input": "",
                "output": item.get("text", ""),
            })

    print(f"Extracted {len(examples)} assistant responses")
    return examples


def download_wizard_evol() -> List[Dict[str, Any]]:
    """Download WizardLM Evol-Instruct dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Install with: pip install datasets")
        sys.exit(1)

    print("Downloading WizardLM Evol-Instruct dataset...")
    dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")

    print(f"Downloaded {len(dataset)} examples")

    # Convert to Alpaca format
    examples = []
    for item in dataset:
        # WizardLM has conversations field
        conversations = item.get("conversations", [])

        if len(conversations) >= 2:
            # Extract human question and assistant response
            human_msg = conversations[0].get("value", "") if conversations[0].get("from") == "human" else ""
            assistant_msg = conversations[1].get("value", "") if conversations[1].get("from") == "gpt" else ""

            if human_msg and assistant_msg:
                examples.append({
                    "instruction": human_msg,
                    "input": "",
                    "output": assistant_msg,
                })

    print(f"Extracted {len(examples)} instruction pairs")
    return examples


def download_sharegpt() -> List[Dict[str, Any]]:
    """Download ShareGPT dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Install with: pip install datasets")
        sys.exit(1)

    print("Downloading ShareGPT dataset...")
    # Using a curated version
    dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")

    print(f"Downloaded {len(dataset)} examples")

    # Convert to Alpaca format
    examples = []
    for item in dataset:
        conversations = item.get("conversations", [])

        if len(conversations) >= 2:
            # Get first user-assistant pair
            user_msg = None
            assistant_msg = None

            for i, conv in enumerate(conversations):
                if conv.get("from") in ["human", "user"] and user_msg is None:
                    user_msg = conv.get("value", "")
                elif conv.get("from") in ["gpt", "assistant"] and assistant_msg is None and user_msg:
                    assistant_msg = conv.get("value", "")
                    break

            if user_msg and assistant_msg:
                examples.append({
                    "instruction": user_msg,
                    "input": "",
                    "output": assistant_msg,
                })

    print(f"Extracted {len(examples)} instruction pairs")
    return examples


def sample_dataset(examples: List[Dict[str, Any]], num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample a subset of examples."""
    if num_samples == -1 or num_samples >= len(examples):
        return examples

    import random
    random.seed(seed)

    sampled = random.sample(examples, num_samples)
    print(f"Sampled {len(sampled)} examples from {len(examples)}")

    return sampled


def format_examples(examples: List[Dict[str, Any]], format_type: str) -> List[str]:
    """Format examples according to specified format."""
    formatted = []

    for example in examples:
        if format_type == "alpaca":
            text = format_alpaca_prompt(
                example["instruction"],
                example.get("input", ""),
                example.get("output", "")
            )
        elif format_type == "raw":
            # Just concatenate instruction and output
            text = f"{example['instruction']}\n\n{example.get('output', '')}"
        elif format_type == "chatml":
            # ChatML format
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example.get("output", "")}
            ]
            text = json.dumps({"messages": messages})
        else:
            raise ValueError(f"Unknown format: {format_type}")

        formatted.append(text)

    return formatted


def save_jsonl(examples: List[str], output_path: str):
    """Save examples to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in examples:
            json_line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Saved {len(examples)} examples to: {output_path}")


def create_train_test_split(
    examples: List[str],
    output_base: str,
    split_ratio: float,
    seed: int = 42
):
    """Create train/test split and save to separate files."""
    import random
    random.seed(seed)

    # Shuffle
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * split_ratio)
    train_examples = shuffled[:split_idx]
    test_examples = shuffled[split_idx:]

    # Generate paths
    base_name = os.path.splitext(output_base)[0]
    train_path = f"{base_name}_train.jsonl"
    test_path = f"{base_name}_test.jsonl"

    # Save
    save_jsonl(train_examples, train_path)
    save_jsonl(test_examples, test_path)

    print(f"\nTrain/Test split created:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Test: {test_path} ({len(test_examples)} examples)")


def print_sample(examples: List[str], num_samples: int = 3):
    """Print a few sample examples."""
    print(f"\n{'='*80}")
    print("SAMPLE EXAMPLES")
    print(f"{'='*80}\n")

    for i, example in enumerate(examples[:num_samples], 1):
        print(f"Example {i}:")
        print("-" * 80)
        # Parse JSON to get text
        try:
            data = json.loads(example)
            text = data.get("text", example)
            # Print first 500 characters
            print(text[:500] + ("..." if len(text) > 500 else ""))
        except json.JSONDecodeError:
            print(example[:500] + ("..." if len(example) > 500 else ""))
        print()


def main():
    """Main entry point."""
    args = parse_args()

    # Download dataset
    if args.dataset == "alpaca":
        examples = download_alpaca(cleaned=False)
    elif args.dataset == "alpaca-cleaned":
        examples = download_alpaca(cleaned=True)
    elif args.dataset == "dolly":
        examples = download_dolly()
    elif args.dataset == "oasst1":
        examples = download_oasst1()
    elif args.dataset == "wizard-evol":
        examples = download_wizard_evol()
    elif args.dataset == "sharegpt":
        examples = download_sharegpt()
    elif args.dataset == "custom":
        print("Custom dataset mode - implement your own data loading here")
        sys.exit(1)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    # Sample if requested
    if args.num_samples != -1:
        examples = sample_dataset(examples, args.num_samples, seed=args.seed)

    # Format examples
    formatted_examples = format_examples(examples, args.format)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        num_str = str(args.num_samples) if args.num_samples != -1 else "full"
        output_path = f"data/{args.dataset}_{num_str}_sample.jsonl"

    # Save or split
    if args.train_test_split:
        create_train_test_split(
            formatted_examples,
            output_path,
            args.train_test_split,
            seed=args.seed
        )
    else:
        save_jsonl(formatted_examples, output_path)

    # Print samples
    print_sample(formatted_examples)

    print(f"\n{'='*80}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nDataset: {args.dataset}")
    print(f"Format: {args.format}")
    print(f"Total examples: {len(formatted_examples)}")
    print(f"\nReady for training! Use this dataset with:")
    print(f"  python scripts/run_experiment_suite.py --suite <config.yaml>")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
