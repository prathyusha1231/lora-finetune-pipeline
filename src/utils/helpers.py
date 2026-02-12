"""
Helper utilities for LoRA fine-tuning.
"""

import os
import random
import torch
import numpy as np
from typing import Dict, Any


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model) -> Dict[str, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }


def print_model_info(model):
    """Print model parameter information."""
    params = count_parameters(model)
    print("\nModel Parameters:")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Percent:   {params['trainable_percent']:.2f}%")


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def format_memory_usage() -> str:
    """Format current GPU memory usage."""
    if not torch.cuda.is_available():
        return "GPU not available"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Total: {total:.2f}GB"


def estimate_memory_requirements(
    model_params: int,
    batch_size: int,
    seq_length: int,
    use_4bit: bool = True,
) -> Dict[str, float]:
    """Estimate memory requirements for training."""
    # Rough estimates
    bytes_per_param = 0.5 if use_4bit else 2  # 4-bit vs fp16
    model_memory = model_params * bytes_per_param / 1024**3

    # Activation memory (rough estimate)
    activation_memory = batch_size * seq_length * 4096 * 4 / 1024**3

    # Optimizer states (AdamW: 2 states per param)
    # Only for trainable params with LoRA (~1% of model)
    optimizer_memory = model_params * 0.01 * 8 / 1024**3

    return {
        "model_gb": model_memory,
        "activation_gb": activation_memory,
        "optimizer_gb": optimizer_memory,
        "total_estimated_gb": model_memory + activation_memory + optimizer_memory,
    }
