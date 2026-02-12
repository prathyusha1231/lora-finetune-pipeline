"""
Inference module for running predictions with fine-tuned LoRA models.
"""

import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LoRAInference:
    """Handle inference with LoRA fine-tuned models."""

    def __init__(
        self,
        base_model: str,
        lora_weights: str,
        use_4bit: bool = True,
        device: Optional[str] = None,
    ):
        self.base_model_name = base_model
        self.lora_weights_path = lora_weights
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the base model with LoRA weights."""
        print(f"Loading model from: {self.base_model_name}")
        print(f"Loading LoRA weights from: {self.lora_weights_path}")

        # Quantization config
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)
        self.model.eval()

        return self.model, self.tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt."""
        if self.model is None:
            self.load_model()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from output
        response = generated_text[len(prompt):].strip()

        return response

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, max_new_tokens, **kwargs) for prompt in prompts]


def main():
    """CLI interface for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Model Inference")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--lora_weights", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    # Initialize inference
    inference = LoRAInference(
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        use_4bit=not args.no_4bit,
    )

    # Load model
    inference.load_model()

    # Generate
    response = inference.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\n" + "=" * 50)
    print("Generated Response:")
    print("=" * 50)
    print(response)


if __name__ == "__main__":
    main()
