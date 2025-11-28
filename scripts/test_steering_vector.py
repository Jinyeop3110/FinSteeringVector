#!/usr/bin/env python3
"""
Test steering vector effectiveness on financial QA.

This script evaluates whether applying a steering vector to 0-shot prompts
can improve performance towards N-shot levels.

The steering vector is applied via forward hooks at a specified layer to the
last K tokens (default: 3 for "Reasoning:" = ['Reason', 'ing', ':']):
    h'_token = h_token + alpha * v_steer

Output format matches the standard evaluation output:
- predictions.json: Full predictions with prompts, outputs, and accuracy
- metrics.json: Aggregate metrics by operation type
- comparison.xlsx: Excel file for easy comparison

Usage:
    python test_steering_vector.py --steering_vector steering_vectors/steering_vector_layer16_3shot_4seeds_nocontext.pt
    python test_steering_vector.py --steering_vector steering_vectors/steering_vector_layer16_3shot_4seeds_nocontext.pt --scales 0.0 0.1 0.2 0.5 1.0
"""

import argparse
import torch
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.finqa import FinQADataset
from src.prompts.chain_of_thought import ChainOfThoughtPrompt
from src.evaluate import extract_answer_from_cot, execution_accuracy


class MultiTokenSteeringHook:
    """Hook to add steering vectors at multiple token positions.

    Applies steering to the last K tokens (for left-padded batches).
    This targets the "Reasoning:" tokens: ['Reason', 'ing', ':']
    """

    def __init__(self, steering_vectors: dict, scale: float = 1.0,
                 n_tokens: int = 3, apply_once: bool = True):
        """
        Args:
            steering_vectors: dict with keys 'pos_0', 'pos_1', 'pos_2', etc.
                             or single tensor for backward compatibility
            scale: Scaling factor for steering
            n_tokens: Number of last tokens to steer
            apply_once: Only apply on first forward pass (for generation)
        """
        self.scale = scale
        self.n_tokens = n_tokens
        self.apply_once = apply_once
        self.applied = False
        self.handle = None

        # Handle both dict and single tensor inputs
        if isinstance(steering_vectors, dict):
            self.steering_vectors = steering_vectors
        else:
            # Single vector - apply to all positions
            self.steering_vectors = {f"pos_{i}": steering_vectors for i in range(n_tokens)}

    def __call__(self, module, input, output):
        if self.apply_once and self.applied:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]

            # Apply steering to last n_tokens positions
            for pos in range(self.n_tokens):
                token_idx = -(pos + 1)  # -1, -2, -3, ...
                pos_key = f"pos_{pos}"

                if pos_key in self.steering_vectors:
                    sv = self.steering_vectors[pos_key]
                    sv = sv.to(hidden_states.device, dtype=hidden_states.dtype)
                    hidden_states[:, token_idx, :] = hidden_states[:, token_idx, :] + self.scale * sv

            self.applied = True
            return (hidden_states,) + output[1:]
        else:
            for pos in range(self.n_tokens):
                token_idx = -(pos + 1)
                pos_key = f"pos_{pos}"

                if pos_key in self.steering_vectors:
                    sv = self.steering_vectors[pos_key]
                    sv = sv.to(output.device, dtype=output.dtype)
                    output[:, token_idx, :] = output[:, token_idx, :] + self.scale * sv

            self.applied = True
            return output

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()

    def reset(self):
        self.applied = False


def run_batched_evaluation_with_details(
    model, tokenizer, dataset, examples: List[Dict], prompts: List[str], gold_answers: List[str],
    steering_vectors: dict, layer_module, scale: float, n_tokens: int,
    max_new_tokens: int, batch_size: int, max_length: int = 2048
) -> Tuple[List[Dict], Dict]:
    """Run batched generation with optional multi-token steering, returning detailed results."""

    predictions = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    for batch_idx in tqdm(range(num_batches), desc=f"scale={scale}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_golds = gold_answers[start_idx:end_idx]
        batch_examples = examples[start_idx:end_idx]

        # Tokenize batch with left padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        # Set up hook if steering
        hook = None
        if scale != 0.0:
            hook = MultiTokenSteeringHook(
                steering_vectors, scale=scale, n_tokens=n_tokens, apply_once=True
            )
            hook.register(layer_module)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if hook:
            hook.remove()

        # Decode and evaluate
        for i, (output, gold, example) in enumerate(zip(outputs, batch_golds, batch_examples)):
            input_len = inputs['input_ids'][i].shape[0]
            output_len = output.shape[0] - input_len

            # Count tokens (excluding padding)
            non_pad_input = (inputs['input_ids'][i] != tokenizer.pad_token_id).sum().item()
            total_input_tokens += non_pad_input
            total_output_tokens += output_len

            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            pred = extract_answer_from_cot(response)
            acc = execution_accuracy(pred, str(gold))

            predictions.append({
                "id": example.get("id", f"sample_{start_idx + i}"),
                "idx": start_idx + i,
                "question": example["question"],
                "context": example["context"][:500] + "..." if len(example["context"]) > 500 else example["context"],
                "ground_truth": str(gold),
                "gold_program": example.get("program", ""),
                "full_prompt": batch_prompts[i],
                "raw_output": response,
                "prediction": pred,
                "execution_accuracy": acc
            })

    total_time = time.time() - start_time

    inference_stats = {
        "total_latency_seconds": round(total_time, 2),
        "avg_latency_per_sample_seconds": round(total_time / len(prompts), 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_input_tokens_per_sample": round(total_input_tokens / len(prompts), 1),
        "avg_output_tokens_per_sample": round(total_output_tokens / len(prompts), 1),
        "avg_total_tokens_per_sample": round((total_input_tokens + total_output_tokens) / len(prompts), 1),
        "tokens_per_second": round((total_input_tokens + total_output_tokens) / total_time, 1) if total_time > 0 else 0
    }

    return predictions, inference_stats


def compute_metrics(predictions: List[Dict], dataset) -> Dict:
    """Compute detailed metrics by operation type."""

    # Overall accuracy
    n_correct = sum(1 for p in predictions if p["execution_accuracy"] == 1.0)
    n_samples = len(predictions)

    # By operation type
    op_stats = {
        "add": {"correct": 0, "total": 0},
        "subtract": {"correct": 0, "total": 0},
        "multiply": {"correct": 0, "total": 0},
        "divide": {"correct": 0, "total": 0},
        "exp": {"correct": 0, "total": 0},
        "greater": {"correct": 0, "total": 0},
        "table_sum": {"correct": 0, "total": 0},
        "table_average": {"correct": 0, "total": 0},
        "table_max": {"correct": 0, "total": 0},
        "table_min": {"correct": 0, "total": 0},
        "multi_step": {"correct": 0, "total": 0}
    }

    for pred in predictions:
        program = pred.get("gold_program", "")
        acc = pred["execution_accuracy"]

        # Detect operations in program
        for op in ["add", "subtract", "multiply", "divide", "exp", "greater",
                   "table_sum", "table_average", "table_max", "table_min"]:
            if op in program.lower():
                op_stats[op]["total"] += 1
                if acc == 1.0:
                    op_stats[op]["correct"] += 1

        # Multi-step (3+ operations)
        op_count = program.count(",") + 1 if program else 0
        if op_count >= 3:
            op_stats["multi_step"]["total"] += 1
            if acc == 1.0:
                op_stats["multi_step"]["correct"] += 1

    by_operation = {}
    for op, stats in op_stats.items():
        by_operation[op] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        }

    return {
        "execution_accuracy": n_correct / n_samples if n_samples > 0 else 0.0,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "by_operation": by_operation,
        "by_num_steps": {
            "3_steps": {
                "correct": op_stats["multi_step"]["correct"],
                "total": op_stats["multi_step"]["total"],
                "accuracy": op_stats["multi_step"]["correct"] / op_stats["multi_step"]["total"] if op_stats["multi_step"]["total"] > 0 else 0.0
            }
        }
    }


def create_comparison_excel(predictions: List[Dict], output_path: Path):
    """Create comparison Excel file."""

    df_data = []
    for pred in predictions:
        df_data.append({
            "id": pred["id"],
            "idx": pred["idx"],
            "question": pred["question"],
            "ground_truth": pred["ground_truth"],
            "prediction": pred["prediction"],
            "correct": "✓" if pred["execution_accuracy"] == 1.0 else "✗",
            "raw_output": pred["raw_output"][:500] + "..." if len(pred["raw_output"]) > 500 else pred["raw_output"]
        })

    df = pd.DataFrame(df_data)
    df.to_excel(output_path, index=False, engine='openpyxl')


def main():
    parser = argparse.ArgumentParser(
        description="Test steering vector on financial QA"
    )
    parser.add_argument("--steering_vector", type=str, required=True,
                        help="Path to steering vector file")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to apply steering (default: from file)")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.5, 1.0],
                        help="Scaling factors to test (default: 0, 0.1, 0.2, 0.5, 1.0)")
    parser.add_argument("--n_tokens", type=int, default=3,
                        help="Number of last tokens to steer (default: 3 for 'Reasoning:')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--split", type=str, default="all",
                        help="Dataset split")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45],
                        help="Random seeds for multiple runs (default: 42, 43, 44, 45)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max input sequence length (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="GPU devices to use (default: 0,1,2,3)")
    args = parser.parse_args()

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    base_dir = Path("/home/yeopjin/orcd/pool/workspace/Financial_task_vector")

    print("=" * 70)
    print("Steering Vector Evaluation")
    print("=" * 70)

    # Load steering vector
    sv_path = base_dir / args.steering_vector if not Path(args.steering_vector).is_absolute() else Path(args.steering_vector)
    print(f"Steering vector: {sv_path}")

    sv_data = torch.load(sv_path, weights_only=False)

    # Extract steering vectors - handle both old and new format
    if 'steering_vectors' in sv_data:
        # New multi-token format
        steering_vectors = sv_data['steering_vectors']
        n_tokens_in_file = sv_data.get('n_tokens', 3)
        print(f"Loaded multi-token steering vectors ({n_tokens_in_file} positions)")
    elif 'steering_vector' in sv_data:
        # Old single-token format - replicate to all positions
        sv = sv_data['steering_vector']
        steering_vectors = {f"pos_{i}": sv for i in range(args.n_tokens)}
        print(f"Loaded single steering vector, applying to {args.n_tokens} positions")
    else:
        # Try other key names for backward compatibility
        for key in ['sv_clean', 'sv_reasoning', 'sv_last']:
            if key in sv_data:
                sv = sv_data[key]
                steering_vectors = {f"pos_{i}": sv for i in range(args.n_tokens)}
                print(f"Loaded steering vector from '{key}', applying to {args.n_tokens} positions")
                break
        else:
            raise KeyError(f"Could not find steering vector in keys: {list(sv_data.keys())}")

    # Get layer from file or args
    layer = args.layer if args.layer is not None else sv_data.get('layer', 16)

    # Print steering vector info
    print(f"\nSteering Vector Info:")
    print(f"  Layer: {layer}")
    print(f"  Tokens to steer: last {args.n_tokens}")
    for pos_key, sv in steering_vectors.items():
        if pos_key.startswith("pos_"):
            print(f"  {pos_key} norm: {sv.norm():.4f}")

    print(f"\nEvaluation Settings:")
    print(f"  Scales to test: {args.scales}")
    print(f"  Seeds: {args.seeds} ({len(args.seeds)} runs)")
    print(f"  Split: {args.split}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  GPUs: {args.gpus}")
    print()

    # Load model with multi-GPU support
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Auto-distribute across GPUs
        trust_remote_code=True,
    )
    model.eval()

    # Convert steering vectors to model dtype
    for key in steering_vectors:
        steering_vectors[key] = steering_vectors[key].to(torch.bfloat16)

    # Get layer module
    layer_module = model.model.layers[layer]
    print(f"Hooking layer {layer}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sv_name = sv_path.stem  # e.g., steering_vector_layer16_3shot_4seeds_nocontext
    model_short = args.model.split("/")[-1]

    outputs_dir = base_dir / "outputs_SV"
    outputs_dir.mkdir(exist_ok=True)

    # Track results across all seeds and scales
    # Structure: {scale: {seed: accuracy}}
    all_results = {scale: {} for scale in args.scales}

    # Run evaluation for each seed
    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'#'*70}")
        print(f"SEED {seed} ({seed_idx + 1}/{len(args.seeds)})")
        print("#" * 70)

        # Load dataset with this seed
        dataset = FinQADataset(split=args.split, seed=seed, use_qa_json=True)
        dataset.load()

        n_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

        # Prepare all prompts and examples
        prompt_template = ChainOfThoughtPrompt(n_shots=0)
        prompts = []
        gold_answers = []
        examples = []

        for idx in range(n_samples):
            example = dataset[idx]
            messages = prompt_template.format(question=example["question"], context=example["context"])
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            gold_answers.append(example.get("exe_answer", example.get("answer", "")))
            examples.append(example)

        print(f"Loaded {len(prompts)} samples with seed {seed}")

        # Run evaluation for each scale
        for scale in args.scales:
            print(f"\n{'='*70}")
            print(f"Running seed={seed}, scale={scale}")
            print("=" * 70)

            predictions, inference_stats = run_batched_evaluation_with_details(
                model, tokenizer, dataset, examples, prompts, gold_answers,
                steering_vectors, layer_module, scale, args.n_tokens,
                args.max_new_tokens, args.batch_size, args.max_length
            )

            # Compute metrics
            metrics = compute_metrics(predictions, dataset)
            metrics["inference_stats"] = inference_stats
            metrics["scale"] = scale
            metrics["layer"] = layer
            metrics["seed"] = seed
            metrics["steering_vector"] = str(sv_path.name)

            acc = metrics["execution_accuracy"] * 100
            print(f"  Accuracy: {acc:.2f}% ({metrics['n_correct']}/{metrics['n_samples']})")

            # Store result
            all_results[scale][seed] = acc

            # Create output directory for this scale and seed
            scale_str = f"scale{scale}".replace(".", "_")
            run_dir = outputs_dir / f"sv_{sv_name}_{scale_str}_seed{seed}_{model_short}_{timestamp}"
            run_dir.mkdir(exist_ok=True)

            # Save predictions.json
            with open(run_dir / "predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)

            # Save metrics.json
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Save comparison.xlsx
            create_comparison_excel(predictions, run_dir / "comparison.xlsx")

            # Save config
            config = {
                "steering_vector": str(sv_path),
                "layer": layer,
                "scale": scale,
                "n_tokens": args.n_tokens,
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "n_samples": n_samples,
                "split": args.split,
                "seed": seed,
                "seeds": args.seeds
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

    # Final summary with mean and std across seeds
    import numpy as np

    print("\n" + "=" * 70)
    print("FINAL RESULTS (aggregated across seeds)")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"Split: {args.split}")
    print(f"Layer: {layer}")
    print(f"Tokens steered: last {args.n_tokens}")
    print()

    summary_results = {}
    for scale in args.scales:
        accs = list(all_results[scale].values())
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        label = "baseline (no steering)" if scale == 0.0 else f"steering scale={scale}"
        print(f"  {label}: {mean_acc:.2f}% ± {std_acc:.2f}%  (runs: {accs})")
        summary_results[scale] = {"mean": mean_acc, "std": std_acc, "runs": accs}

    # Save summary
    summary_file = outputs_dir / f"summary_{sv_name}_{model_short}_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "steering_vector": str(sv_path),
            "layer": layer,
            "n_tokens": args.n_tokens,
            "seeds": args.seeds,
            "results": summary_results
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    print(f"Outputs saved to: {outputs_dir}")


if __name__ == "__main__":
    main()
