# Available HuggingFace Datasets

This project supports multiple high-quality instruction-following datasets from HuggingFace. All datasets are automatically downloaded and formatted for training.

## Quick Start

```bash
# Download any dataset with 1000 samples
python scripts/prepare_dataset.py --dataset <name> --num-samples 1000
```

## Supported Datasets

### 1. Alpaca (Recommended for Quick Experiments)

**Dataset**: `alpaca` or `alpaca-cleaned`

**Source**: Stanford / Community
**Size**: 52,000 examples
**Quality**: ⭐⭐⭐⭐ (Good)
**Best For**: Quick experiments, baseline comparisons

**Description**: Stanford's instruction-following dataset generated from GPT-3.5. The cleaned version removes duplicates and low-quality examples.

**Usage**:
```bash
# Original version
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000

# Cleaned version (recommended)
python scripts/prepare_dataset.py --dataset alpaca-cleaned --num-samples 1000
```

**Format**: Instruction + Optional Input + Output

**Example**:
```
Instruction: What is the capital of France?
Output: The capital of France is Paris.
```

---

### 2. Dolly (High Quality, Curated)

**Dataset**: `dolly`

**Source**: Databricks
**Size**: 15,000 examples
**Quality**: ⭐⭐⭐⭐⭐ (Excellent)
**Best For**: High-quality training, production models

**Description**: Hand-curated instruction dataset created by Databricks employees. Higher quality than Alpaca but smaller.

**Usage**:
```bash
python scripts/prepare_dataset.py --dataset dolly --num-samples 1000
```

**Format**: Instruction + Context + Response

**Why Use It**: Fewer examples but better quality. Good for testing if quality > quantity.

---

### 3. OpenAssistant (OASST1) (Conversational)

**Dataset**: `oasst1`

**Source**: OpenAssistant / LAION
**Size**: ~160,000 messages (conversation threads)
**Quality**: ⭐⭐⭐⭐⭐ (Excellent)
**Best For**: Conversational AI, multi-turn dialogue

**Description**: Community-created conversation dataset with human-rated responses. Very high quality, natural conversations.

**Usage**:
```bash
python scripts/prepare_dataset.py --dataset oasst1 --num-samples 1000
```

**Format**: Conversational threads (we extract user-assistant pairs)

**Why Use It**: Best for building conversational models. Responses are ranked by humans.

---

### 4. WizardLM Evol-Instruct (Complex Instructions)

**Dataset**: `wizard-evol`

**Source**: WizardLM Team
**Size**: 196,000 examples
**Quality**: ⭐⭐⭐⭐ (Very Good)
**Best For**: Complex reasoning, advanced instructions

**Description**: Instructions evolved from simple to complex using GPT-4. Tests model's ability to follow complex, multi-step instructions.

**Usage**:
```bash
python scripts/prepare_dataset.py --dataset wizard-evol --num-samples 1000
```

**Format**: Evolved instructions with detailed responses

**Why Use It**: Best for training models on complex reasoning tasks. Good for benchmarking.

---

### 5. ShareGPT (Real ChatGPT Conversations)

**Dataset**: `sharegpt`

**Source**: Community (shared ChatGPT conversations)
**Size**: ~90,000 conversations
**Quality**: ⭐⭐⭐⭐ (Very Good)
**Best For**: ChatGPT-style responses, natural dialogue

**Description**: Real conversations shared by ChatGPT users. Diverse topics and natural language.

**Usage**:
```bash
python scripts/prepare_dataset.py --dataset sharegpt --num-samples 1000
```

**Format**: Multi-turn conversations (we extract first user-assistant pair)

**Why Use It**: Most similar to ChatGPT's training data. Natural and diverse.

---

## Comparison Table

| Dataset | Size | Quality | Speed | Best Use Case |
|---------|------|---------|-------|---------------|
| **Alpaca** | 52K | Good | Fast | Quick experiments, baselines |
| **Alpaca-cleaned** | 52K | Very Good | Fast | Recommended default |
| **Dolly** | 15K | Excellent | Fast | Production, high quality |
| **OASST1** | 160K | Excellent | Medium | Conversational AI |
| **WizardLM** | 196K | Very Good | Slow | Complex reasoning |
| **ShareGPT** | 90K | Very Good | Medium | ChatGPT-style models |

## Recommendations by Use Case

### For This LoRA Efficiency Study

**Best Choice**: `alpaca-cleaned` (1000-2000 samples)

**Why**:
- Fast to download and process
- Good quality for comparing LoRA configurations
- Well-established baseline
- Enough examples to see differences

**Command**:
```bash
python scripts/prepare_dataset.py --dataset alpaca-cleaned --num-samples 1000
```

### For Production Models

**Best Choice**: `dolly` or `oasst1` (full dataset or 5000+ samples)

**Why**:
- Higher quality responses
- Better for real-world applications
- Human-curated or human-rated

**Command**:
```bash
python scripts/prepare_dataset.py --dataset dolly --num-samples 5000
python scripts/prepare_dataset.py --dataset oasst1 --num-samples 5000
```

### For Research / Benchmarking

**Best Choice**: `wizard-evol` (1000+ samples)

**Why**:
- Complex instructions test model capabilities
- Good for ablation studies
- Shows model limits

**Command**:
```bash
python scripts/prepare_dataset.py --dataset wizard-evol --num-samples 1000
```

### For ChatGPT-like Models

**Best Choice**: `sharegpt` (2000+ samples)

**Why**:
- Real ChatGPT conversations
- Natural dialogue patterns
- Diverse topics

**Command**:
```bash
python scripts/prepare_dataset.py --dataset sharegpt --num-samples 2000
```

## Dataset Sizes for Different Experiments

### Quick Test (Minutes)
```bash
--num-samples 100
```
Perfect for: Testing pipeline, debugging

### Small Experiment (30-60 min)
```bash
--num-samples 1000
```
Perfect for: LoRA rank comparisons, quick ablations

### Medium Experiment (2-4 hours)
```bash
--num-samples 5000
```
Perfect for: Serious experiments, publishable results

### Full Training (4-8 hours)
```bash
--num-samples -1  # All data
```
Perfect for: Production models, best results

## Advanced Usage

### Train/Test Split

```bash
# 90% train, 10% test
python scripts/prepare_dataset.py \
    --dataset alpaca-cleaned \
    --num-samples 1000 \
    --train-test-split 0.9
```

This creates:
- `data/alpaca-cleaned_1000_sample_train.jsonl` (900 examples)
- `data/alpaca-cleaned_1000_sample_test.jsonl` (100 examples)

### Multiple Datasets

Combine different datasets for diversity:

```bash
# Download Alpaca
python scripts/prepare_dataset.py --dataset alpaca-cleaned --num-samples 500 --output data/mix_alpaca.jsonl

# Download Dolly
python scripts/prepare_dataset.py --dataset dolly --num-samples 500 --output data/mix_dolly.jsonl

# Combine them (manual)
cat data/mix_alpaca.jsonl data/mix_dolly.jsonl > data/mixed_dataset.jsonl
```

### Custom Format

Change output format:

```bash
# Alpaca format (default)
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000 --format alpaca

# Raw text (simpler)
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000 --format raw

# ChatML format
python scripts/prepare_dataset.py --dataset alpaca --num-samples 1000 --format chatml
```

## Troubleshooting

### Download Issues

```bash
# If download fails, try using HuggingFace Hub login
pip install huggingface_hub
huggingface-cli login

# Then retry
python scripts/prepare_dataset.py --dataset <name> --num-samples 1000
```

### Slow Downloads

Large datasets (wizard-evol, sharegpt) may take time:
```bash
# Use smaller sample first to test
python scripts/prepare_dataset.py --dataset wizard-evol --num-samples 100

# Then increase if working
python scripts/prepare_dataset.py --dataset wizard-evol --num-samples 1000
```

### Memory Issues

If dataset processing uses too much RAM:
```bash
# Process in smaller batches by using smaller num-samples
python scripts/prepare_dataset.py --dataset sharegpt --num-samples 500
```

## Dataset License Notes

- **Alpaca**: Research use (based on GPT-3.5 outputs)
- **Dolly**: CC-BY-SA-3.0 (commercial use OK)
- **OASST1**: Apache 2.0 (commercial use OK)
- **WizardLM**: Research use
- **ShareGPT**: Check specific version license

Always verify license compatibility for your use case.

## Next Steps

After preparing a dataset:

1. **Update experiment configs** to use your dataset:
   ```yaml
   dataset_path: "data/alpaca-cleaned_1000_sample.jsonl"
   ```

2. **Run experiments**:
   ```bash
   python scripts/run_experiment_suite.py --suite configs/experiments/rank_sweep.yaml
   ```

3. **Evaluate results**:
   ```bash
   python scripts/generate_report.py
   ```

---

**Recommendation for this project**: Start with `alpaca-cleaned` (1000 samples) for fast iteration and reliable results.
