"""
Profiling tools for memory and speed measurements.
"""

import time
import torch
from typing import Optional, Dict


class MemoryProfiler:
    """Profile GPU memory usage during model operations."""

    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.device_available = torch.cuda.is_available()

    def start(self):
        """Start memory profiling."""
        if self.device_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB

    def get_current_memory(self) -> float:
        """Get current memory usage in GB."""
        if self.device_available:
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB since last start()."""
        if self.device_available:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            return peak
        return 0.0

    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        if not self.device_available:
            return {"error": "CUDA not available"}

        torch.cuda.synchronize()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
        }


class SpeedProfiler:
    """Profile execution speed and throughput."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds."""
        self.end_time = time.time()
        return self.end_time - self.start_time

    def get_elapsed(self) -> Optional[float]:
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return None
        return time.time() - self.start_time


class InferenceProfiler:
    """Profile inference latency and throughput."""

    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.speed_profiler = SpeedProfiler()

    def profile_inference(
        self,
        model,
        tokenizer,
        prompts: list,
        max_new_tokens: int = 50,
        num_warmup: int = 2,
    ) -> Dict[str, float]:
        """
        Profile model inference performance.

        Args:
            model: The language model
            tokenizer: Tokenizer
            prompts: List of prompts to test
            max_new_tokens: Max tokens to generate
            num_warmup: Number of warmup runs

        Returns:
            Dictionary with latency and throughput metrics
        """
        model.eval()

        # Warmup
        for i in range(min(num_warmup, len(prompts))):
            inputs = tokenizer(prompts[i], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Actual profiling
        latencies = []
        memory_usage = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            self.memory_profiler.start()
            self.speed_profiler.start()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            latency = self.speed_profiler.stop()
            memory = self.memory_profiler.get_peak_memory()

            latencies.append(latency)
            memory_usage.append(memory)

            # Count generated tokens
            num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        import numpy as np

        return {
            "avg_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "avg_memory_gb": np.mean(memory_usage),
            "tokens_per_second": max_new_tokens / np.mean(latencies) if latencies else 0,
            "num_samples": len(prompts),
        }
