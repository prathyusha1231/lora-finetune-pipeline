"""
Visualization utilities for experiment results.
Generates publication-quality plots for LoRA efficiency study.
"""

import os
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from pathlib import Path


class ExperimentVisualizer:
    """
    Generate visualizations from experiment results.

    Creates various plots to analyze LoRA efficiency tradeoffs:
    - Rank comparison charts
    - Efficiency frontier plots
    - Training time comparisons
    - Memory usage analysis
    """

    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style for professional-looking plots
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        self.colors = plt.cm.tab10.colors

    def plot_rank_comparison(
        self,
        experiments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot comparison of different LoRA ranks.

        Shows accuracy and memory usage across different ranks.

        Args:
            experiments: List of experiment results (from database)
            output_path: Path to save plot (default: output_dir/rank_comparison.png)

        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "rank_comparison.png")

        # Extract data
        ranks = []
        perplexities = []
        memory_usage = []

        for exp in experiments:
            # Get rank from config
            config = exp.get("config", {})
            params = config.get("parameters", {})
            rank = params.get("lora_rank")

            if rank is None:
                continue

            # Get metrics
            metrics = exp.get("metrics", {})
            profiling = exp.get("profiling", {})

            perplexity = metrics.get("perplexity")
            memory = profiling.get("peak_training_memory_gb")

            if perplexity and memory:
                ranks.append(rank)
                perplexities.append(perplexity)
                memory_usage.append(memory)

        # Sort by rank
        sorted_data = sorted(zip(ranks, perplexities, memory_usage))
        ranks, perplexities, memory_usage = zip(*sorted_data)

        # Create figure with dual y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot perplexity (lower is better)
        color = self.colors[0]
        ax1.set_xlabel('LoRA Rank', fontsize=12)
        ax1.set_ylabel('Perplexity (lower is better)', color=color, fontsize=12)
        line1 = ax1.plot(ranks, perplexities, 'o-', color=color, linewidth=2, markersize=8, label='Perplexity')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Plot memory usage on secondary axis
        ax2 = ax1.twinx()
        color = self.colors[1]
        ax2.set_ylabel('Peak Memory (GB)', color=color, fontsize=12)
        line2 = ax2.plot(ranks, memory_usage, 's--', color=color, linewidth=2, markersize=8, label='Memory')
        ax2.tick_params(axis='y', labelcolor=color)

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Title
        plt.title('LoRA Rank Ablation: Accuracy vs Memory Tradeoff', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved rank comparison plot: {output_path}")
        return output_path

    def plot_efficiency_frontier(
        self,
        experiments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot accuracy vs memory efficiency frontier.

        Scatter plot showing the tradeoff between model accuracy and memory usage.

        Args:
            experiments: List of experiment results
            output_path: Path to save plot

        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "efficiency_frontier.png")

        # Extract data
        memory_usage = []
        perplexities = []
        labels = []
        colors_list = []

        experiment_types = {}

        for exp in experiments:
            config = exp.get("config", {})
            exp_type = config.get("experiment_type", "unknown")
            exp_name = config.get("experiment_name", "unknown")

            metrics = exp.get("metrics", {})
            profiling = exp.get("profiling", {})

            perplexity = metrics.get("perplexity")
            memory = profiling.get("peak_training_memory_gb")

            if perplexity and memory:
                memory_usage.append(memory)
                perplexities.append(perplexity)
                labels.append(exp_name)

                # Color by experiment type
                if exp_type not in experiment_types:
                    experiment_types[exp_type] = len(experiment_types)
                colors_list.append(self.colors[experiment_types[exp_type] % len(self.colors)])

        # Create scatter plot
        plt.figure(figsize=(10, 6))

        for memory, perplexity, label, color in zip(memory_usage, perplexities, labels, colors_list):
            plt.scatter(memory, perplexity, s=100, c=[color], alpha=0.7, edgecolors='black', linewidth=1)

        plt.xlabel('Peak Memory Usage (GB)', fontsize=12)
        plt.ylabel('Perplexity (lower is better)', fontsize=12)
        plt.title('Efficiency Frontier: Accuracy vs Memory', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Add legend for experiment types
        legend_elements = [
            plt.scatter([], [], s=100, c=[self.colors[idx % len(self.colors)]],
                       alpha=0.7, edgecolors='black', linewidth=1, label=exp_type)
            for exp_type, idx in experiment_types.items()
        ]
        plt.legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved efficiency frontier plot: {output_path}")
        return output_path

    def plot_training_time_comparison(
        self,
        experiments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot training time comparison across experiments.

        Horizontal bar chart showing training times.

        Args:
            experiments: List of experiment results
            output_path: Path to save plot

        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "training_time_comparison.png")

        # Extract data
        names = []
        times = []

        for exp in experiments:
            config = exp.get("config", {})
            profiling = exp.get("profiling", {})

            name = config.get("experiment_name", "unknown")
            time_seconds = profiling.get("training_time_seconds")

            if time_seconds:
                names.append(name)
                times.append(time_seconds / 60)  # Convert to minutes

        # Sort by time
        sorted_data = sorted(zip(names, times), key=lambda x: x[1])
        names, times = zip(*sorted_data) if sorted_data else ([], [])

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, times, color=self.colors[2], alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(time + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{time:.1f} min',
                   va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Training Time (minutes)', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved training time comparison plot: {output_path}")
        return output_path

    def plot_model_size_comparison(
        self,
        experiments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot model size comparison across experiments.

        Bar chart showing LoRA adapter sizes.

        Args:
            experiments: List of experiment results
            output_path: Path to save plot

        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "model_size_comparison.png")

        # Extract data
        names = []
        sizes = []

        for exp in experiments:
            config = exp.get("config", {})
            profiling = exp.get("profiling", {})

            name = config.get("experiment_name", "unknown")
            size_mb = profiling.get("model_size_mb")

            if size_mb:
                names.append(name)
                sizes.append(size_mb)

        # Sort by size
        sorted_data = sorted(zip(names, sizes), key=lambda x: x[1])
        names, sizes = zip(*sorted_data) if sorted_data else ([], [])

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, sizes, color=self.colors[4], alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            ax.text(size + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{size:.1f} MB',
                   va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Model Size (MB)', fontsize=12)
        ax.set_title('LoRA Adapter Size Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved model size comparison plot: {output_path}")
        return output_path

    def plot_parameters_comparison(
        self,
        experiments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot trainable parameters comparison.

        Shows the number of trainable parameters vs total parameters.

        Args:
            experiments: List of experiment results
            output_path: Path to save plot

        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "parameters_comparison.png")

        # Extract data
        names = []
        trainable = []
        total = []

        for exp in experiments:
            config = exp.get("config", {})
            profiling = exp.get("profiling", {})

            name = config.get("experiment_name", "unknown")
            trainable_params = profiling.get("trainable_params")
            total_params = profiling.get("total_params")

            if trainable_params and total_params:
                names.append(name)
                trainable.append(trainable_params / 1e6)  # Convert to millions
                total.append(total_params / 1e6)

        # Create grouped bar chart
        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width/2, total, width, label='Total Params', color=self.colors[0], alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, trainable, width, label='Trainable Params', color=self.colors[1], alpha=0.7, edgecolor='black')

        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Parameters (Millions)', fontsize=12)
        ax.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved parameters comparison plot: {output_path}")
        return output_path

    def generate_all_plots(self, experiments: List[Dict[str, Any]]) -> List[str]:
        """
        Generate all available plots for experiments.

        Args:
            experiments: List of experiment results

        Returns:
            List of paths to generated plots
        """
        print(f"\nGenerating visualizations for {len(experiments)} experiments...")

        plots = []

        # Filter experiments by type for specific plots
        rank_experiments = [e for e in experiments if e.get("config", {}).get("experiment_type") == "rank_ablation"]

        if rank_experiments:
            plots.append(self.plot_rank_comparison(rank_experiments))

        # General plots for all experiments
        if len(experiments) > 1:
            plots.append(self.plot_efficiency_frontier(experiments))
            plots.append(self.plot_training_time_comparison(experiments))
            plots.append(self.plot_model_size_comparison(experiments))
            plots.append(self.plot_parameters_comparison(experiments))

        print(f"\nGenerated {len(plots)} plots in {self.output_dir}")
        return plots
