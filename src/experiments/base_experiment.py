"""
Base experiment class for LoRA efficiency studies.
Provides abstract interface for reproducible experiments.
"""

import os
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from transformers import TrainingArguments

from ..train import LoRATrainingConfig, LoRATrainer
from ..evaluation.metrics import MetricsCalculator
from ..evaluation.profiler import MemoryProfiler, SpeedProfiler


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    experiment_name: str
    experiment_type: str  # 'rank_ablation', 'module_ablation', 'quantization'
    base_model: str
    dataset_path: str
    output_dir: str

    # Experiment-specific parameters (override in subclasses)
    parameters: Dict[str, Any] = None

    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    max_seq_length: int = 512
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class ExperimentResult:
    """Container for experiment results."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics = {}
        self.profiling = {}
        self.metadata = {}
        self.start_time = None
        self.end_time = None

    def add_metric(self, name: str, value: float):
        """Add a metric to results."""
        self.metrics[name] = value

    def add_profiling_data(self, data: Dict[str, Any]):
        """Add profiling data."""
        self.profiling.update(data)

    def add_metadata(self, key: str, value: Any):
        """Add metadata."""
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "profiling": self.profiling,
            "metadata": self.metadata,
            "duration_seconds": (self.end_time - self.start_time) if self.end_time else None,
            "timestamp": self.start_time,
        }

    def save(self, path: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.

    Subclasses must implement:
    - get_lora_config(): Return LoRA configuration for this experiment
    - get_experiment_name(): Return unique experiment identifier
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result = ExperimentResult(config)
        self.trainer = None
        self.metrics_calculator = MetricsCalculator()
        self.memory_profiler = MemoryProfiler()
        self.speed_profiler = SpeedProfiler()

    @abstractmethod
    def get_lora_config(self) -> Dict[str, Any]:
        """
        Return LoRA-specific configuration for this experiment.
        Must include: lora_r, lora_alpha, target_modules, etc.
        """
        pass

    @abstractmethod
    def get_experiment_name(self) -> str:
        """Return a unique name for this experiment."""
        pass

    def setup_training_config(self) -> LoRATrainingConfig:
        """Create training configuration with experiment-specific LoRA settings."""
        lora_config = self.get_lora_config()

        return LoRATrainingConfig(
            base_model=self.config.base_model,
            output_dir=os.path.join(self.config.output_dir, self.get_experiment_name()),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            max_seq_length=self.config.max_seq_length,
            **lora_config
        )

    def run(self) -> ExperimentResult:
        """
        Execute the full experiment pipeline.

        Returns:
            ExperimentResult with all metrics and profiling data
        """
        print(f"\n{'='*80}")
        print(f"Running Experiment: {self.get_experiment_name()}")
        print(f"{'='*80}\n")

        self.result.start_time = time.time()

        try:
            # 1. Setup
            print("Setting up training configuration...")
            training_config = self.setup_training_config()
            self.trainer = LoRATrainer(training_config)

            # 2. Load model with memory profiling
            print("Loading model...")
            self.memory_profiler.start()
            model, tokenizer = self.trainer.load_model()
            model_load_memory = self.memory_profiler.get_peak_memory()
            self.result.add_profiling_data({
                "model_load_memory_gb": model_load_memory,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in model.parameters()),
            })

            # 3. Prepare dataset
            print("Preparing dataset...")
            dataset = self.trainer.prepare_dataset(
                self.config.dataset_path,
                text_column="text"
            )
            self.result.add_metadata("dataset_size", len(dataset))

            # 4. Training with speed profiling
            print("Starting training...")
            self.memory_profiler.start()
            self.speed_profiler.start()

            self.trainer.train(dataset)

            training_time = self.speed_profiler.stop()
            peak_memory = self.memory_profiler.get_peak_memory()

            self.result.add_profiling_data({
                "training_time_seconds": training_time,
                "peak_training_memory_gb": peak_memory,
                "tokens_per_second": self._calculate_throughput(dataset, training_time),
            })

            # 5. Evaluation
            print("Evaluating model...")
            eval_metrics = self.evaluate_model(model, tokenizer, dataset)
            for name, value in eval_metrics.items():
                self.result.add_metric(name, value)

            # 6. Model size on disk
            model_size = self._get_model_size_on_disk(training_config.output_dir)
            self.result.add_profiling_data({"model_size_mb": model_size})

            print(f"\n{'='*80}")
            print(f"Experiment Complete: {self.get_experiment_name()}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            self.result.add_metadata("error", str(e))
            self.result.add_metadata("status", "failed")
            raise
        finally:
            self.result.end_time = time.time()

        self.result.add_metadata("status", "completed")
        return self.result

    def evaluate_model(self, model, tokenizer, dataset) -> Dict[str, float]:
        """
        Evaluate the trained model.
        Override in subclasses for custom evaluation.
        """
        # Use last 10% as test set
        test_size = len(dataset) // 10
        test_dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))

        metrics = self.metrics_calculator.compute_perplexity(
            model, tokenizer, test_dataset
        )

        return metrics

    def _calculate_throughput(self, dataset, training_time: float) -> float:
        """Calculate training throughput in tokens/second."""
        total_tokens = len(dataset) * self.config.max_seq_length * self.config.num_train_epochs
        return total_tokens / training_time if training_time > 0 else 0

    def _get_model_size_on_disk(self, model_dir: str) -> float:
        """Calculate total model size in MB."""
        total_size = 0
        final_model_path = os.path.join(model_dir, "final_model")

        if os.path.exists(final_model_path):
            for dirpath, dirnames, filenames in os.walk(final_model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)

        return total_size / (1024 * 1024)  # Convert to MB

    def save_results(self, output_path: Optional[str] = None):
        """Save experiment results to file."""
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                self.get_experiment_name(),
                "results.json"
            )

        self.result.save(output_path)
        print(f"Results saved to: {output_path}")
