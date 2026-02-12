"""
LoRA target module ablation experiments.
Tests different combinations of attention layers to apply LoRA to.
"""

from typing import Dict, Any, List
from .base_experiment import BaseExperiment, ExperimentConfig


class ModuleAblationExperiment(BaseExperiment):
    """
    Experiment to test different target module combinations.

    Common configurations:
    - Q+V only: ["q_proj", "v_proj"]
    - Q+K+V: ["q_proj", "k_proj", "v_proj"]
    - Q+K+V+O: ["q_proj", "k_proj", "v_proj", "o_proj"]
    - All attention + MLP: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    """

    def __init__(self, config: ExperimentConfig, target_modules: List[str], variant_name: str = None):
        super().__init__(config)
        self.target_modules = target_modules
        self.variant_name = variant_name or "_".join(sorted(target_modules))

        # Store in config parameters
        if config.parameters is None:
            config.parameters = {}
        config.parameters["target_modules"] = target_modules

    def get_lora_config(self) -> Dict[str, Any]:
        """Return LoRA configuration with specified target modules."""
        return {
            "lora_r": 16,  # Keep rank constant
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": self.target_modules,
            "use_4bit": True,
        }

    def get_experiment_name(self) -> str:
        """Return unique experiment identifier."""
        return f"module_ablation_{self.variant_name}"


# Predefined module configurations
MODULE_CONFIGS = {
    "qv_only": ["q_proj", "v_proj"],
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_full": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
