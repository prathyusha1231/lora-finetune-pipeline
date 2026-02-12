"""
Metrics calculation for model evaluation.
"""

import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


class MetricsCalculator:
    """Calculate evaluation metrics for language models."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_perplexity(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Compute perplexity on a dataset.

        Args:
            model: The language model
            tokenizer: Tokenizer
            dataset: HuggingFace dataset with 'input_ids' and 'attention_mask'
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with perplexity and loss
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        dataloader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                loss = outputs.loss
                num_tokens = attention_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)

        return {
            "perplexity": float(perplexity),
            "loss": float(avg_loss),
            "total_tokens_evaluated": total_tokens,
        }

    def compute_accuracy(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        expected_outputs: List[str],
        max_new_tokens: int = 50,
    ) -> Dict[str, float]:
        """
        Compute accuracy on Q&A style tasks.

        Args:
            model: The language model
            tokenizer: Tokenizer
            prompts: List of input prompts
            expected_outputs: List of expected responses
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with accuracy metrics
        """
        model.eval()
        correct = 0
        total = len(prompts)

        with torch.no_grad():
            for prompt, expected in zip(prompts, expected_outputs):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                )

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = generated[len(prompt):].strip()

                # Simple exact match (can be made more sophisticated)
                if expected.lower() in generated.lower():
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def compute_generation_metrics(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> Dict[str, float]:
        """
        Compute generation quality metrics.

        Args:
            model: The language model
            tokenizer: Tokenizer
            prompts: List of input prompts
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with generation metrics
        """
        model.eval()
        response_lengths = []
        unique_tokens_ratio = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.7,
                    do_sample=True,
                )

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated[len(prompt):].strip()

                # Track response length
                response_lengths.append(len(response.split()))

                # Track unique token ratio (diversity measure)
                tokens = response.split()
                if len(tokens) > 0:
                    unique_ratio = len(set(tokens)) / len(tokens)
                    unique_tokens_ratio.append(unique_ratio)

        return {
            "avg_response_length": np.mean(response_lengths) if response_lengths else 0,
            "avg_unique_token_ratio": np.mean(unique_tokens_ratio) if unique_tokens_ratio else 0,
            "num_samples": len(prompts),
        }
