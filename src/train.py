"""
Main training module for LoRA fine-tuning.
Supports various base models and custom datasets.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
import yaml


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning."""

    # Model settings
    base_model: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_name: Optional[str] = None

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Quantization
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Output
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 100

    @classmethod
    def from_yaml(cls, path: str) -> "LoRATrainingConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class LoRATrainer:
    """Handles the LoRA fine-tuning process."""

    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Configure quantization settings."""
        if self.config.use_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def load_model(self):
        """Load and prepare the base model with LoRA."""
        print(f"Loading base model: {self.config.base_model}")

        # Setup quantization
        bnb_config = self.setup_quantization()

        # Load tokenizer
        tokenizer_name = self.config.tokenizer_name or self.config.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def prepare_dataset(self, dataset_path: str, text_column: str = "text"):
        """Load and tokenize the dataset."""
        print(f"Loading dataset from: {dataset_path}")

        # Load dataset (supports local files or HuggingFace datasets)
        if os.path.exists(dataset_path):
            if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            elif dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=dataset_path, split="train")
            else:
                dataset = load_dataset(dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")

        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenized_dataset

    def train(self, dataset):
        """Run the training process."""
        print("Starting training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_32bit",
            report_to="none",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train
        self.trainer.train()

        # Save the final model
        self.save_model()

        return self.trainer

    def save_model(self, path: Optional[str] = None):
        """Save the LoRA adapters."""
        save_path = path or os.path.join(self.config.output_dir, "final_model")
        print(f"Saving model to: {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--text_column", type=str, default="text", help="Name of text column in dataset")
    args = parser.parse_args()

    # Load config
    config = LoRATrainingConfig.from_yaml(args.config)

    # Initialize trainer
    trainer = LoRATrainer(config)

    # Load model
    trainer.load_model()

    # Prepare dataset
    dataset = trainer.prepare_dataset(args.dataset, args.text_column)

    # Train
    trainer.train(dataset)

    print("Training complete!")


if __name__ == "__main__":
    main()
