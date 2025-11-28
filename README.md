# FinTaskVector

> Can we compress few-shot learning into a steering vector?

This project evaluates Chain-of-Thought prompting strategies for financial numerical reasoning on the [FinQA dataset](https://github.com/czyssrs/FinQA), and investigates whether few-shot benefits can be distilled into steering vectors.

## Key Results

| Method | Accuracy | Input Tokens | vs 0-shot |
|--------|----------|--------------|-----------|
| Vanilla 0-shot | 8.39% | 1,179 | - |
| CoT 0-shot | 29.14% | 1,215 | baseline |
| **CoT 0-shot + FSV** | **31.00%** | **1,215** | **+1.86%** |
| CoT 3-shot | 32.08% | 5,257 | +2.94% |

**Key Finding**: Steering vectors (layer 12, scale=0.2) recover **63% of the 3-shot gain** with **zero extra tokens**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FinQA dataset
bash scripts/download_finqa.sh

# 3. Run best prompting config (CoT 3-shot)
python run.py --config configs/qwen2.5-1.5b/cot_3shot_answer.yaml

# 4. Or run with steering vector (no extra tokens needed)
python scripts/test_steering_vector.py \
    --steering_vector steering_vectors/layer12_3shot.pt \
    --scales 0.2
```

## Documentation

| Resource | Description |
|----------|-------------|
| [Technical Report](report/report.md) | Full methodology, results, and analysis |
| [Configs](configs/) | YAML configurations for all experiments |

---

## Project Structure

```
FinTaskVector/
├── run.py                      # Main evaluation script
├── configs/                    # Experiment configurations
│   └── qwen2.5-1.5b/          # Model-specific configs
├── scripts/
│   ├── download_finqa.sh      # Dataset download
│   ├── extract_steering_vector.py
│   └── test_steering_vector.py
├── outputs/                    # Evaluation results
├── steering_vectors/           # Extracted FSV files
└── report/                     # Technical report
```

## Dataset

**FinQA**: Financial QA requiring multi-step numerical reasoning over tables and text.

| Split | Examples |
|-------|----------|
| Train | 6,251 |
| Dev | 883 |
| Test | 1,147 |

## Model

Default: `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, 1.5B parameters)

Inference: [vLLM](https://github.com/vllm-project/vllm) for batch processing, HuggingFace Transformers for steering vector experiments.

---

## Detailed Usage

### Prompting Evaluation

| Config | Command |
|--------|---------|
| Vanilla 0-shot | `python run.py --config configs/qwen2.5-1.5b/vanilla_0shot_answer.yaml` |
| CoT 0-shot | `python run.py --config configs/qwen2.5-1.5b/cot_0shot_answer.yaml` |
| CoT 3-shot | `python run.py --config configs/qwen2.5-1.5b/cot_3shot_answer.yaml` |

Results saved to `outputs/{tag}_{model}_{timestamp}/`.

### Steering Vector Pipeline

**Step 1: Extract** steering vectors from hidden state differences:

```bash
python scripts/extract_steering_vector.py \
    --layers 12 16 \
    --n_shots 3 \
    --seeds 42 43 44 45
```

**Step 2: Test** steering vectors at different scales:

```bash
python scripts/test_steering_vector.py \
    --steering_vector steering_vectors/layer12_3shot.pt \
    --scales 0.0 0.1 0.2 0.5 1.0
```

### Output Format

```
outputs_SV/
  sv_{name}_{scale}_seed{S}/
    predictions.json   # Per-sample predictions
    metrics.json       # Accuracy stats
    config.json        # Run configuration
  summary.json         # Aggregated results across seeds
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{fintaskvector2024,
  title={Chain-of-Thought Prompting for Financial Numerical Reasoning},
  author={...},
  year={2024}
}
```

## License

MIT
