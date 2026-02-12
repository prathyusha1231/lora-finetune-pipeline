# Quick Start Guide

This guide will get you running experiments in minutes.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

## Option 1: Run a Single Quick Experiment (No GPU Required for Setup)

```bash
# 1. Prepare a small dataset
python scripts/prepare_dataset.py --dataset alpaca --num-samples 100

# 2. Test the framework (will fail without GPU, but validates setup)
python scripts/run_single_experiment.py \
    --experiment-type rank_ablation \
    --lora-rank 8 \
    --dataset data/alpaca_100_sample.jsonl \
    --base-model microsoft/phi-2 \
    --num-epochs 1 \
    --dry-run  # Add this to preview without running
```

## Option 2: Run Full Experiment Suite (GPU Required)

```bash
# 1. Prepare dataset (1000 samples for faster experiments)
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000

# 2. Run rank ablation suite (tests ranks 4, 8, 16, 32, 64)
python scripts/run_experiment_suite.py \
    --suite configs/experiments/rank_sweep.yaml

# 3. Run module ablation suite
python scripts/run_experiment_suite.py \
    --suite configs/experiments/module_sweep.yaml

# 4. Run quantization study
python scripts/run_experiment_suite.py \
    --suite configs/experiments/quantization_sweep.yaml

# 5. Generate comprehensive report
python scripts/generate_report.py
```

Expected time with Phi-2 (2.7B) on 4-bit:
- Each rank experiment: ~5-10 minutes
- Full rank suite (5 experiments): ~45-60 minutes
- Module suite (6 experiments): ~50-70 minutes
- Quantization suite (2 experiments): ~15-20 minutes
- **Total: ~2-3 hours for all experiments**

## Option 3: Preview Without Running (No GPU)

You can explore the framework without running experiments:

```bash
# Preview what experiments would run
python scripts/run_experiment_suite.py \
    --suite configs/experiments/rank_sweep.yaml \
    --dry-run

# Examine the experiment configurations
cat configs/experiments/rank_sweep.yaml
cat configs/experiments/module_sweep.yaml
cat configs/experiments/quantization_sweep.yaml

# Explore the code structure
ls -R src/
```

## Common Workflows

### Test on Small Model (Fastest)

```bash
# Use Phi-2 (2.7B) - fits on most consumer GPUs
python scripts/run_experiment_suite.py \
    --suite configs/experiments/rank_sweep.yaml
```

### Production Results (Slower, Better)

Edit `configs/experiments/rank_sweep.yaml`:
```yaml
base_model: "meta-llama/Llama-2-7b-hf"  # Or "mistralai/Mistral-7B-v0.1"
num_train_epochs: 3
```

Then run the suite.

### Custom Experiment

```bash
python scripts/run_single_experiment.py \
    --experiment-type rank_ablation \
    --lora-rank 16 \
    --dataset data/your_dataset.jsonl \
    --base-model microsoft/phi-2 \
    --num-epochs 3 \
    --batch-size 4
```

## Viewing Results

### During Experiments

Results are automatically logged to:
- SQLite database: `results/experiments.db`
- JSON files: `output/<experiment_name>/results.json`

### After Experiments

```bash
# Generate markdown report with visualizations
python scripts/generate_report.py

# View the report
cat RESULTS.md

# Check plots
ls results/plots/
```

### Query Results Programmatically

```python
from src.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
experiments = tracker.get_all_experiments(status="completed")

for exp in experiments:
    print(f"{exp['experiment_name']}: {exp['metrics']['perplexity']:.2f}")
```

## Evaluating Trained Models

```bash
# Evaluate a specific checkpoint
python scripts/evaluate_model.py \
    --checkpoint output/rank_ablation_r16/final_model \
    --base-model microsoft/phi-2 \
    --dataset data/alpaca_test.jsonl

# Run benchmark prompts
python scripts/evaluate_model.py \
    --checkpoint output/rank_ablation_r16/final_model \
    --base-model microsoft/phi-2 \
    --benchmark-only

# Profile inference performance
python scripts/evaluate_model.py \
    --checkpoint output/rank_ablation_r16/final_model \
    --base-model microsoft/phi-2 \
    --profile-inference
```

## Troubleshooting

### Out of Memory

1. Reduce batch size in config:
   ```yaml
   per_device_train_batch_size: 2  # Instead of 4
   ```

2. Use smaller model:
   ```yaml
   base_model: "microsoft/phi-2"  # 2.7B instead of 7B
   ```

3. Reduce sequence length:
   ```yaml
   max_seq_length: 256  # Instead of 512
   ```

### Dataset Not Found

```bash
# Make sure to run prepare_dataset.py first
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000

# Check the file was created
ls data/
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# If using conda
conda install pytorch transformers datasets -c pytorch
```

## Memory Requirements

### Minimum (Phi-2, 4-bit, rank=8)
- VRAM: 6-8GB
- RAM: 16GB
- GPU: RTX 3060, GTX 1660 Ti, or better

### Recommended (Phi-2, 4-bit, rank=16)
- VRAM: 8-12GB
- RAM: 32GB
- GPU: RTX 3070, RTX 4060, or better

### Best Results (Llama-2-7B, 4-bit, rank=16)
- VRAM: 16-20GB
- RAM: 32GB+
- GPU: RTX 3090, RTX 4090, A100

## Next Steps

1. **Run experiments**: Start with rank ablation suite
2. **Generate report**: Use `generate_report.py`
3. **Analyze results**: Review RESULTS.md and plots
4. **Iterate**: Adjust configurations based on findings
5. **Document**: Update README with your specific results

## Getting Help

- Check `README.md` for detailed documentation
- Review `PROJECT_STATUS_AND_ROADMAP.txt` for project overview
- Examine example configs in `configs/experiments/`
- Read the code - it's well-commented!

## Tips for Success

1. **Start small**: Test with 100 samples first
2. **Use 4-bit**: Minimal quality loss, huge memory savings
3. **Monitor GPU**: Use `nvidia-smi` to watch memory usage
4. **Save checkpoints**: Experiments save automatically to `output/`
5. **Track results**: Everything goes to the database for analysis

Happy experimenting! 🚀
