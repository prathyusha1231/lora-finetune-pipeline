"""
LoRA rank ablation experiments.
Tests different LoRA rank values to study the accuracy-memory tradeoff.
"""

from typing import Dict, Any
from .base_experiment import BaseExperiment, ExperimentConfig


class RankAblationExperiment(BaseExperiment):
    """
    Experiment to test different LoRA rank values.

    Tests ranks: 4, 8, 16, 32, 64
    Keeps all other hyperparameters constant.
    """

    def __init__(self, config: ExperimentConfig, lora_rank: int):
        super().__init__(config)
        self.lora_rank = lora_rank

        # Store in config parameters
        if config.parameters is None:
            config.parameters = {}
        config.parameters["lora_rank"] = lora_rank

    def get_lora_config(self) -> Dict[str, Any]:
        """Return LoRA configuration with specified rank."""
        return {
            "lora_r": self.lora_rank,
            "lora_alpha": self.lora_rank * 2,  # Standard: alpha = 2 * rank
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "use_4bit": True,  # Keep quantization constant
        }

    def get_experiment_name(self) -> str:
        """Return unique experiment identifier."""
        return f"rank_ablation_r{self.lora_rank}"
