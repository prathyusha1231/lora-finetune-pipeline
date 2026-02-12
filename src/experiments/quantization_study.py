"""
Quantization efficiency experiments.
Compares 4-bit, 8-bit, and full precision training.
"""

from typing import Dict, Any
from .base_experiment import BaseExperiment, ExperimentConfig


class QuantizationExperiment(BaseExperiment):
    """
    Experiment to compare different quantization strategies.

    Modes:
    - 4bit: 4-bit NF4 quantization (lowest memory)
    - 8bit: 8-bit quantization (medium memory)
    - fp16: Full precision float16 (highest memory)
    """

    def __init__(self, config: ExperimentConfig, quantization_mode: str):
        """
        Args:
            config: Experiment configuration
            quantization_mode: One of '4bit', '8bit', 'fp16'
        """
        super().__init__(config)

        valid_modes = ['4bit', '8bit', 'fp16']
        if quantization_mode not in valid_modes:
            raise ValueError(f"quantization_mode must be one of {valid_modes}")

        self.quantization_mode = quantization_mode

        # Store in config parameters
        if config.parameters is None:
            config.parameters = {}
        config.parameters["quantization_mode"] = quantization_mode

    def get_lora_config(self) -> Dict[str, Any]:
        """Return LoRA configuration with specified quantization."""
        base_config = {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }

        if self.quantization_mode == "4bit":
            base_config.update({
                "use_4bit": True,
                "use_8bit": False,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
            })
        elif self.quantization_mode == "8bit":
            base_config.update({
                "use_4bit": False,
                "use_8bit": True,
            })
        else:  # fp16
            base_config.update({
                "use_4bit": False,
                "use_8bit": False,
            })

        return base_config

    def get_experiment_name(self) -> str:
        """Return unique experiment identifier."""
        return f"quantization_{self.quantization_mode}"
