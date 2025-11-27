# FinTaskVector

Evaluation framework for financial question answering using different prompting strategies.

## Summary

This project evaluates LLM performance on financial numerical reasoning tasks using the FinQA dataset. It compares zero-shot, few-shot (in-context learning), and chain-of-thought prompting strategies.

## Dataset

- **FinQA** ([czyssrs/FinQA](https://github.com/czyssrs/FinQA)): Financial QA dataset with numerical reasoning over financial documents
  - Train: 6,251 examples
  - Dev: 883 examples
  - Test: 1,147 examples

Download the dataset:
```bash
bash scripts/download_finqa.sh
```

## Models

Default model: `Qwen/Qwen2.5-7B-Instruct`

Inference powered by [vLLM](https://github.com/vllm-project/vllm) for efficient batch processing.

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Run Evaluation

Zero-shot:
```bash
python run.py --config configs/zero_shot.yaml
```

Few-shot (3 examples):
```bash
python run.py --config configs/few_shot.yaml
```

Custom configuration:
```bash
python run.py --model Qwen/Qwen2.5-7B-Instruct --prompt_type few_shot --n_shots 5
```

### Prompting Strategies

| Strategy | Description |
|----------|-------------|
| `vanilla` | Zero-shot, no examples |
| `few_shot` | N in-context examples |
| `cot` | Chain-of-thought reasoning |

Results are saved to `outputs/` with predictions and metrics.
