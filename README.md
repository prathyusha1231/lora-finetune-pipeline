# LoRA Efficiency Study

A production-ready ML engineering project for systematic analysis of LoRA (Low-Rank Adaptation) hyperparameters. This project demonstrates rigorous experimental methodology, automated evaluation, and professional ML engineering practices.

## Why This Project Exists

Unlike typical "I fine-tuned a model" projects, this is a **systematic efficiency study** that answers critical questions about LoRA fine-tuning:

- How does LoRA rank affect model accuracy vs memory usage?
- Which attention modules are most important to fine-tune?
- What's the performance difference between 4-bit, 8-bit, and FP16 training?
- What's the optimal configuration for limited GPU memory?

This project demonstrates:
- Deep understanding of LoRA mechanics and tradeoffs
- Rigorous experimental methodology with reproducible results
- Production-quality Python code (not Jupyter notebooks)
- Automated evaluation and benchmarking pipelines
- Systematic performance analysis with visualizations

## Key Features

- **Automated Experiment Suites**: Run multiple experiments with different configurations automatically
- **Built-in Evaluation**: Comprehensive metrics including perplexity, memory profiling, and speed benchmarks
- **Result Tracking**: SQLite database for experiment results with queryable history
- **Automated Reporting**: Generate markdown reports with visualizations
- **Production Code**: Clean, well-structured Python with proper CLI tools
- **Memory Optimized**: 4-bit and 8-bit quantization via bitsandbytes
- **Flexible Configuration**: YAML-based configs for reproducible experiments

## Project Structure

```
lora-finetune-pipeline/
├── src/
│   ├── train.py                      # Core training logic
│   ├── inference.py                  # Inference utilities
│   ├── data/
│   │   └── dataset.py                # Dataset processing
│   ├── experiments/                  # Experiment framework
│   │   ├── base_experiment.py        # Abstract experiment class
│   │   ├── rank_ablation.py          # Rank ablation experiments
│   │   ├── module_ablation.py        # Module ablation experiments
│   │   └── quantization_study.py     # Quantization experiments
│   ├── evaluation/                   # Evaluation suite
│   │   ├── metrics.py                # Perplexity, accuracy metrics
│   │   ├── profiler.py               # Memory & speed profiling
│   │   └── benchmarks.py             # Benchmark task suite
│   └── utils/
│       ├── helpers.py                # Helper functions
│       ├── experiment_tracker.py     # SQLite experiment tracking
│       └── visualization.py          # Plot generation
├── scripts/
│   ├── prepare_dataset.py            # Dataset preparation tool
│   ├── run_single_experiment.py      # Run one experiment
│   ├── run_experiment_suite.py       # Run experiment suite
│   ├── evaluate_model.py             # Evaluate trained models
│   └── generate_report.py            # Generate analysis report
├── configs/
│   ├── default_config.yaml           # Default training config
│   └── experiments/                  # Experiment suite configs
│       ├── rank_sweep.yaml           # Rank ablation suite
│       ├── module_sweep.yaml         # Module ablation suite
│       └── quantization_sweep.yaml   # Quantization study
├── data/                             # Datasets (gitignored)
├── output/                           # Trained models (gitignored)
├── results/                          # Experiment results
│   ├── experiments.db                # SQLite results database
│   └── plots/                        # Generated visualizations
├── requirements.txt
├── README.md
└── RESULTS.md                        # Generated analysis report
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lora-finetune-pipeline.git
cd lora-finetune-pipeline

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for visualization
pip install matplotlib seaborn
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (8GB+ VRAM for 4-bit, 16GB+ for 8-bit)
- PyTorch 2.0+
- See `requirements.txt` for full list

## Quick Start

### 1. Prepare Dataset

Download and format the Alpaca dataset (or use your own):

```bash
# Download 1000-sample dataset for quick experiments
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000

# Or download full dataset
python scripts/prepare_dataset.py --dataset alpaca --num-samples -1
```

This creates `data/alpaca_1000_sample.jsonl` ready for training.

### 2. Run a Single Experiment

Test a specific configuration:

```bash
# Run rank ablation experiment with rank=16
python scripts/run_single_experiment.py \
    --experiment-type rank_ablation \
    --lora-rank 16 \
    --dataset data/alpaca_1000_sample.jsonl \
    --base-model microsoft/phi-2
```

### 3. Run Experiment Suite

Run multiple experiments automatically:

```bash
# Run rank ablation suite (tests ranks: 4, 8, 16, 32, 64)
python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml

# Run module ablation suite
python scripts/run_experiment_suite.py --suite configs/experiments/module_sweep.yaml

# Run quantization study
python scripts/run_experiment_suite.py --suite configs/experiments/quantization_sweep.yaml
```

### 4. Generate Analysis Report

After running experiments, generate a comprehensive report:

```bash
python scripts/generate_report.py
```

This creates:
- `RESULTS.md` - Detailed analysis with findings and recommendations
- `results/plots/` - Visualizations (rank comparison, efficiency frontier, etc.)

## Experiment Types

### Rank Ablation

Tests different LoRA rank values (4, 8, 16, 32, 64) to understand the accuracy-memory tradeoff.

**Configuration**: `configs/experiments/rank_sweep.yaml`

**Key Questions**:
- How does rank affect model accuracy?
- What's the memory overhead of higher ranks?
- Where are the diminishing returns?

### Module Ablation

Tests different combinations of target modules (Q, K, V, O projections, MLP layers).

**Configuration**: `configs/experiments/module_sweep.yaml`

**Key Questions**:
- Which modules are most important?
- Can we skip some layers for efficiency?
- Is MLP fine-tuning necessary?

### Quantization Study

Compares 4-bit, 8-bit, and FP16 quantization modes.

**Configuration**: `configs/experiments/quantization_sweep.yaml`

**Key Questions**:
- How much accuracy do we lose with 4-bit?
- Is 8-bit worth the extra memory?
- What's the speed-accuracy tradeoff?

## CLI Tools

### prepare_dataset.py

Download and format datasets for training.

```bash
# Alpaca dataset (1000 samples)
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000

# With train/test split
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000 --train-test-split 0.9
```

### run_single_experiment.py

Run a single experiment with specific configuration.

```bash
python scripts/run_single_experiment.py \
    --experiment-type rank_ablation \
    --lora-rank 16 \
    --dataset data/alpaca_1000_sample.jsonl \
    --base-model microsoft/phi-2 \
    --num-epochs 3
```

### run_experiment_suite.py

Run multiple experiments from a YAML configuration.

```bash
# Run suite
python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml

# Continue on errors
python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml --continue-on-error

# Dry run (preview without running)
python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml --dry-run
```

### evaluate_model.py

Evaluate a trained model checkpoint.

```bash
# Evaluate on test dataset
python scripts/evaluate_model.py \
    --checkpoint output/rank_ablation_r16/final_model \
    --base-model microsoft/phi-2 \
    --dataset data/alpaca_test.jsonl

# Run benchmark prompts
python scripts/evaluate_model.py \
    --checkpoint output/rank_ablation_r16/final_model \
    --base-model microsoft/phi-2 \
    --benchmark-only

# Profile inference speed
python scripts/evaluate_model.py \
    --checkpoint output/rank_ablation_r16/final_model \
    --base-model microsoft/phi-2 \
    --profile-inference
```

### generate_report.py

Generate comprehensive analysis report from experiments.

```bash
# Generate full report
python scripts/generate_report.py

# Filter by experiment type
python scripts/generate_report.py --experiment-type rank_ablation

# Custom output
python scripts/generate_report.py --output reports/my_analysis.md
```

## Typical Workflow

1. **Prepare dataset**:
   ```bash
   python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000
   ```

2. **Run experiments**:
   ```bash
   python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml
   python scripts/run_experiment_suite.py --suite configs/experiments/module_sweep.yaml
   python scripts/run_experiment_suite.py --suite configs/experiments/quantization_sweep.yaml
   ```

3. **Generate report**:
   ```bash
   python scripts/generate_report.py
   ```

4. **Review results**:
   - Read `RESULTS.md` for detailed analysis
   - Check `results/plots/` for visualizations
   - Query `results/experiments.db` for raw data

## Configuration

### Experiment Suite YAML Format

```yaml
suite_name: "my_experiment_suite"
base_model: "microsoft/phi-2"
dataset_path: "data/alpaca_sample.jsonl"
num_train_epochs: 2
per_device_train_batch_size: 4
max_seq_length: 512

experiments:
  - name: "exp1"
    lora_rank: 16
  - name: "exp2"
    lora_rank: 32
```

## Memory Requirements

Approximate VRAM usage with 4-bit quantization (Phi-2 2.7B model):

| LoRA Rank | Training | Inference |
|-----------|----------|-----------|
| 4         | ~6GB     | ~4GB      |
| 8         | ~7GB     | ~4GB      |
| 16        | ~8GB     | ~4GB      |
| 32        | ~12GB    | ~5GB      |
| 64        | ~16GB    | ~6GB      |

For larger models (7B):

| Model Size | 4-bit Training | 8-bit Training | FP16 Training |
|------------|----------------|----------------|---------------|
| 7B         | ~16GB          | ~24GB          | ~40GB         |
| 13B        | ~24GB          | ~40GB          | ~60GB+        |

## Experiment Results Database

All experiment results are stored in `results/experiments.db`. You can query it directly:

```python
from src.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Get all completed experiments
experiments = tracker.get_all_experiments(status="completed")

# Get experiments by type
rank_experiments = tracker.get_experiments_by_type("rank_ablation")

# Export to JSON
tracker.export_to_json("my_experiments.json")
```

## Visualization

The visualization system generates publication-quality plots:

- **Rank Comparison**: Accuracy vs memory across different ranks
- **Efficiency Frontier**: Scatter plot of accuracy vs memory
- **Training Time Comparison**: Bar chart of training times
- **Model Size Comparison**: LoRA adapter sizes
- **Parameters Comparison**: Trainable vs total parameters

All plots are automatically generated by `generate_report.py`.

## Extending the Framework

### Add a New Experiment Type

1. Create a new experiment class in `src/experiments/`:

```python
from .base_experiment import BaseExperiment, ExperimentConfig

class MyExperiment(BaseExperiment):
    def get_lora_config(self):
        return {
            "lora_r": 16,
            "lora_alpha": 32,
            # Your custom config
        }

    def get_experiment_name(self):
        return "my_experiment"
```

2. Create a config file in `configs/experiments/my_sweep.yaml`

3. Run with `run_experiment_suite.py`

## Tips for Best Results

1. **Start Small**: Use microsoft/phi-2 (2.7B) for fast iteration
2. **Use 4-bit**: Minimal accuracy loss, huge memory savings
3. **Rank 16 is Good Default**: Balance of accuracy and efficiency
4. **Target Attention First**: Q+K+V+O gives best bang for buck
5. **Monitor Memory**: Use profiling to optimize for your GPU

## Contributing

Contributions are welcome! This is a portfolio/research project, but improvements to the experimental framework are appreciated.

## Citation

If you use this experimental framework in your research:

```bibtex
@misc{lora-efficiency-study,
  title={LoRA Efficiency Study: Systematic Analysis of Hyperparameter Tradeoffs},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lora-finetune-pipeline}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning library
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit/8-bit quantization
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Transformer models
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca) - Instruction-following dataset

## Contact

For questions or discussions about this research:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Note**: This is a research/educational project demonstrating ML engineering best practices. Results may vary based on hardware, dataset, and specific use case.
