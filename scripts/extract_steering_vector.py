#!/usr/bin/env python3
"""
Extract position-aligned steering vectors for financial QA.

This script extracts hidden state representations from 0-shot and N-shot prompts,
using GLOBAL LEFT-PADDING alignment to ensure tokens are at the same absolute positions
(eliminating RoPE positional encoding confounds).

GLOBAL PADDING STRATEGY:
- All samples (both 0-shot and n-shot) are padded to the SAME global max length
- This global max length is determined by the longest n-shot sequence in the dataset
- With left-padding, the "Reasoning:" tokens are at the exact same absolute positions
  across ALL samples, regardless of their original lengths
- This is crucial for position-aligned representation extraction

Uses TRUE DATA PARALLELISM: each GPU runs its own model instance and processes
different seeds in parallel, then results are aggregated.

Supports multiple random seeds to get different ICL demonstration orderings,
aggregating hidden states across all seeds for a more robust steering vector.

Extracts the last K tokens (default: 3 for "Reasoning:" = ['Reason', 'ing', ':'])
and computes steering vectors for each position.

The steering vector is computed as: v_steer = mean(h_nshot) - mean(h_0shot)

Usage:
    python extract_steering_vector.py  # Uses defaults: 4 seeds, layers 16,20, 4 GPUs
    python extract_steering_vector.py --seeds 42 43 44 45 --layers 16 --gpus 0,1,2,3
"""

import argparse
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import sys
import torch.multiprocessing as mp
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import FinQADataset
from src.prompts import ChainOfThoughtPrompt


def extract_hidden_states_for_seed(
    model,
    tokenizer,
    layer: int,
    n_shots: int,
    seed: int,
    n_tokens: int = 3,
    batch_size: int = 8,
    max_length: int = 8192,
    max_samples: int = None,
):
    """
    Extract hidden states for a single seed.

    Returns:
        tuple of (all_hidden_0shot, all_hidden_nshot, n_valid, n_skipped, avg_lens)
    """
    # Re-initialize dataset with this seed
    dataset_seed = FinQADataset(split="all", seed=seed, use_qa_json=True)
    dataset_seed.load()

    n_samples = len(dataset_seed) if max_samples is None else min(max_samples, len(dataset_seed))

    # Create prompt templates
    # For N-shot: use include_context_in_examples=False to get only question+reasoning+answer
    # This isolates the "reasoning pattern" from the specific financial context
    prompt_0shot = ChainOfThoughtPrompt(n_shots=0)
    prompt_nshot = ChainOfThoughtPrompt(n_shots=n_shots, include_context_in_examples=False)

    # Pre-generate all prompts
    all_texts_0shot = []
    all_texts_nshot = []
    all_sample_ids = []
    all_lens_0shot = []
    all_lens_nshot = []
    skipped = 0

    for idx in range(n_samples):
        example = dataset_seed[idx]
        sample_id = example.get("id", f"sample_{idx}")
        icl_examples = dataset_seed.get_icl_examples(n_shots, idx)

        messages_0shot = prompt_0shot.format(
            question=example["question"],
            context=example["context"],
            icl_examples=None,
        )
        messages_nshot = prompt_nshot.format(
            question=example["question"],
            context=example["context"],
            icl_examples=icl_examples,
        )

        text_0shot = tokenizer.apply_chat_template(
            messages_0shot, tokenize=False, add_generation_prompt=True
        )
        text_nshot = tokenizer.apply_chat_template(
            messages_nshot, tokenize=False, add_generation_prompt=True
        )

        # Check lengths before adding
        len_0shot = len(tokenizer.encode(text_0shot, add_special_tokens=False))
        len_nshot = len(tokenizer.encode(text_nshot, add_special_tokens=False))

        if len_nshot > max_length:
            skipped += 1
            continue

        all_texts_0shot.append(text_0shot)
        all_texts_nshot.append(text_nshot)
        all_sample_ids.append((idx, sample_id))
        all_lens_0shot.append(len_0shot)
        all_lens_nshot.append(len_nshot)

    n_valid = len(all_texts_0shot)

    # GLOBAL PADDING: Use fixed max_length for ALL samples
    # This ensures all samples (both 0-shot and n-shot) are padded to the SAME length
    # so that the last N tokens are at the exact same absolute positions across all samples
    global_max_len = max_length  # Fixed global padding length (default: 8024)
    actual_max_len = max(all_lens_nshot) if all_lens_nshot else 0
    print(f"    Global padding length: {global_max_len} tokens (fixed)")
    print(f"    Actual max n-shot length: {actual_max_len} tokens")
    print(f"    (All {n_valid} samples will be left-padded to {global_max_len} tokens)")

    # No need to sort by length anymore since all samples use the same padding
    # But keeping sorted order can still help with cache efficiency
    sorted_indices = np.argsort(all_lens_nshot)

    # Storage for hidden states
    all_hidden_0shot = {pos: [] for pos in range(n_tokens)}
    all_hidden_nshot = {pos: [] for pos in range(n_tokens)}

    # Process in batches
    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        batch_indices = sorted_indices[batch_start:batch_end]

        # Get batch texts (interleave 0-shot and n-shot)
        batch_texts = []
        for i in batch_indices:
            batch_texts.append(all_texts_0shot[i])
            batch_texts.append(all_texts_nshot[i])

        # GLOBAL PADDING: Use global_max_len for ALL batches
        # Both 0-shot and n-shot are padded to the same global length
        # With left padding, the last N tokens are at positions [-1, -2, -3, ...]
        # which are now at the SAME absolute position for ALL samples

        # Tokenize batch with global padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            max_length=global_max_len,  # Same length for ALL batches
            truncation=True,
        ).to(model.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden states (layer+1 because index 0 is embeddings)
        layer_hidden = outputs.hidden_states[layer + 1]

        # Process each sample in batch
        for i, batch_idx in enumerate(batch_indices):
            idx_0shot = i * 2
            idx_nshot = i * 2 + 1

            # Extract last n_tokens positions
            for pos in range(n_tokens):
                token_idx = -(pos + 1)  # -1, -2, -3, ...
                hidden_0shot = layer_hidden[idx_0shot, token_idx, :].cpu().float()
                hidden_nshot = layer_hidden[idx_nshot, token_idx, :].cpu().float()

                all_hidden_0shot[pos].append(hidden_0shot)
                all_hidden_nshot[pos].append(hidden_nshot)

        # Clear CUDA cache
        del outputs, layer_hidden, inputs
        torch.cuda.empty_cache()

    avg_lens = {
        "avg_len_0shot": sum(all_lens_0shot) / len(all_lens_0shot) if all_lens_0shot else 0,
        "avg_len_nshot": sum(all_lens_nshot) / len(all_lens_nshot) if all_lens_nshot else 0,
    }

    return all_hidden_0shot, all_hidden_nshot, n_valid, skipped, avg_lens


def worker_process(
    gpu_id: int,
    seeds: List[int],
    layers: List[int],
    model_name: str,
    n_shots: int,
    n_tokens: int,
    batch_size: int,
    max_length: int,
    max_samples: int,
    output_dir: str,
    result_queue: mp.Queue,
):
    """Worker process that runs on a single GPU."""
    try:
        # Set this process to use only the assigned GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 is our GPU

        print(f"[GPU {gpu_id}] Loading model...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()

        print(f"[GPU {gpu_id}] Model loaded. Processing seeds: {seeds}")

        results = {}

        for layer in layers:
            layer_results = {
                "hidden_0shot": {pos: [] for pos in range(n_tokens)},
                "hidden_nshot": {pos: [] for pos in range(n_tokens)},
                "total_valid": 0,
                "total_skipped": 0,
                "avg_lens": [],
            }

            for seed in tqdm(seeds, desc=f"[GPU {gpu_id}] Layer {layer}"):
                hidden_0shot, hidden_nshot, n_valid, n_skipped, avg_lens = extract_hidden_states_for_seed(
                    model=model,
                    tokenizer=tokenizer,
                    layer=layer,
                    n_shots=n_shots,
                    seed=seed,
                    n_tokens=n_tokens,
                    batch_size=batch_size,
                    max_length=max_length,
                    max_samples=max_samples,
                )

                # Aggregate
                for pos in range(n_tokens):
                    layer_results["hidden_0shot"][pos].extend(hidden_0shot[pos])
                    layer_results["hidden_nshot"][pos].extend(hidden_nshot[pos])

                layer_results["total_valid"] += n_valid
                layer_results["total_skipped"] += n_skipped
                layer_results["avg_lens"].append(avg_lens)

            results[layer] = layer_results

        # Save results to file instead of passing through queue (avoids pickle size limits)
        output_file = Path(output_dir) / f"worker_gpu{gpu_id}_results.pt"
        torch.save(results, output_file)

        # Signal completion via queue (only pass small data)
        result_queue.put((gpu_id, str(output_file)))
        print(f"[GPU {gpu_id}] Done! Saved to {output_file}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((gpu_id, None))


def compute_steering_vectors(all_hidden_0shot, all_hidden_nshot, n_tokens):
    """Compute steering vectors from aggregated hidden states."""
    steering_vectors = {}
    stats = {}

    for pos in range(n_tokens):
        hidden_0shot_all = torch.stack(all_hidden_0shot[pos])
        hidden_nshot_all = torch.stack(all_hidden_nshot[pos])

        mean_0shot = hidden_0shot_all.mean(dim=0)
        mean_nshot = hidden_nshot_all.mean(dim=0)
        sv = mean_nshot - mean_0shot

        pos_name = f"pos_{pos}"
        steering_vectors[pos_name] = sv
        stats[pos_name] = {
            "mean_0shot_norm": mean_0shot.norm().item(),
            "mean_nshot_norm": mean_nshot.norm().item(),
            "steering_vector_norm": sv.norm().item(),
        }

        print(f"  Position -{pos+1}: SV norm = {sv.norm():.4f}")

    # Combined steering vector
    combined_sv = torch.stack([steering_vectors[f"pos_{p}"] for p in range(n_tokens)]).mean(dim=0)
    steering_vectors["combined"] = combined_sv
    stats["combined"] = {"steering_vector_norm": combined_sv.norm().item()}
    print(f"  Combined: SV norm = {combined_sv.norm():.4f}")

    # PCA analysis on last token
    hidden_0shot_last = torch.stack(all_hidden_0shot[0])
    hidden_nshot_last = torch.stack(all_hidden_nshot[0])
    combined = torch.cat([hidden_0shot_last, hidden_nshot_last], dim=0)
    combined_centered = combined - combined.mean(dim=0)
    U, S, Vt = torch.linalg.svd(combined_centered, full_matrices=False)
    var_explained_pc1 = (S[0]**2 / (S**2).sum()).item() * 100

    return steering_vectors, stats, var_explained_pc1


def main():
    parser = argparse.ArgumentParser(
        description="Extract position-aligned steering vectors for financial QA"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20],
                        help="Layers to extract from (default: 16, 20)")
    parser.add_argument("--n_shots", type=int, default=3,
                        help="Number of ICL examples (default: 3)")
    parser.add_argument("--n_tokens", type=int, default=3,
                        help="Number of last tokens to extract (default: 3 for 'Reasoning:')")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45],  # 4 seeds
                        help="Random seeds for ICL sampling (default: 42-45, 4 seeds)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference (default: 4, reduced for global padding)")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Max sequence length (global padding length)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process per seed")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="GPU devices to use (default: 0,1,2,3)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    n_seeds = len(args.seeds)

    base_dir = Path("/home/yeopjin/orcd/pool/workspace/Financial_task_vector")
    steering_vectors_dir = base_dir / "steering_vectors"
    steering_vectors_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Steering Vector Extraction (TRUE DATA PARALLELISM)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Layers: {args.layers}")
    print(f"N-shots: {args.n_shots}")
    print(f"N-tokens: {args.n_tokens} (last {args.n_tokens} tokens)")
    print(f"Seeds: {n_seeds} seeds ({args.seeds[0]}-{args.seeds[-1]})")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs: {gpu_ids} ({n_gpus} GPUs)")
    print(f"Max length: {args.max_length}")
    print()
    print(f"Expected samples: 429 × {n_seeds} seeds = {429 * n_seeds} paired representations")
    print(f"Seeds per GPU: {n_seeds // n_gpus} (+ remainder distributed)")
    print()

    # Distribute seeds across GPUs
    seeds_per_gpu = []
    for i in range(n_gpus):
        start_idx = i * (n_seeds // n_gpus)
        end_idx = (i + 1) * (n_seeds // n_gpus) if i < n_gpus - 1 else n_seeds
        seeds_per_gpu.append(args.seeds[start_idx:end_idx])

    for i, (gpu_id, seeds) in enumerate(zip(gpu_ids, seeds_per_gpu)):
        print(f"  GPU {gpu_id}: seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)")

    # Create temp directory for worker outputs
    temp_dir = steering_vectors_dir / "temp_worker_outputs"
    temp_dir.mkdir(exist_ok=True)

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    # Launch worker processes
    processes = []
    for gpu_id, seeds in zip(gpu_ids, seeds_per_gpu):
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                seeds,
                args.layers,
                args.model,
                args.n_shots,
                args.n_tokens,
                args.batch_size,
                args.max_length,
                args.max_samples,
                str(temp_dir),
                result_queue,
            )
        )
        p.start()
        processes.append(p)

    # Collect results from all workers
    print("\nWaiting for workers to complete...")
    worker_files = {}
    for _ in range(n_gpus):
        gpu_id, result_file = result_queue.get()
        if result_file is not None:
            worker_files[gpu_id] = result_file
            print(f"  Received results from GPU {gpu_id}: {result_file}")
        else:
            print(f"  GPU {gpu_id} failed!")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("\nAll workers completed. Loading and aggregating results...")

    # Load results from worker files
    all_results = {}
    for gpu_id, filepath in worker_files.items():
        print(f"  Loading results from GPU {gpu_id}...")
        all_results[gpu_id] = torch.load(filepath, weights_only=False)

    # Aggregate results across GPUs for each layer
    for layer in args.layers:
        print(f"\n{'='*70}")
        print(f"Aggregating Layer {layer}")
        print("=" * 70)

        aggregated_hidden_0shot = {pos: [] for pos in range(args.n_tokens)}
        aggregated_hidden_nshot = {pos: [] for pos in range(args.n_tokens)}
        total_valid = 0
        total_skipped = 0
        all_avg_lens = []

        for gpu_id in sorted(all_results.keys()):
            layer_results = all_results[gpu_id][layer]

            for pos in range(args.n_tokens):
                aggregated_hidden_0shot[pos].extend(layer_results["hidden_0shot"][pos])
                aggregated_hidden_nshot[pos].extend(layer_results["hidden_nshot"][pos])

            total_valid += layer_results["total_valid"]
            total_skipped += layer_results["total_skipped"]
            all_avg_lens.extend(layer_results["avg_lens"])

        print(f"Total samples aggregated: {total_valid}")
        print(f"Total skipped: {total_skipped}")

        # Compute steering vectors from aggregated data
        print("\nComputing steering vectors from aggregated data...")
        steering_vectors, stats, var_explained_pc1 = compute_steering_vectors(
            aggregated_hidden_0shot, aggregated_hidden_nshot, args.n_tokens
        )

        # Average lengths across seeds
        avg_len_0shot = np.mean([x["avg_len_0shot"] for x in all_avg_lens])
        avg_len_nshot = np.mean([x["avg_len_nshot"] for x in all_avg_lens])

        result = {
            "steering_vectors": steering_vectors,
            "stats": stats,
            "layer": layer,
            "n_shots": args.n_shots,
            "n_tokens": args.n_tokens,
            "n_samples": total_valid,
            "n_skipped": total_skipped,
            "n_seeds": n_seeds,
            "seeds": args.seeds,
            "pc1_variance_explained": var_explained_pc1,
            "avg_len_0shot": avg_len_0shot,
            "avg_len_nshot": avg_len_nshot,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "method": "global_padding_position_aligned_multi_token_data_parallel",
            "padding_strategy": "global",  # All samples padded to same global max length
            "n_gpus": n_gpus,
            "icl_context_mode": "no_context",  # ICL examples have no table/text context
            # Backward compatibility
            "steering_vector": steering_vectors["pos_0"],
        }

        # Save steering vector - add "nocontext" to filename to distinguish from full-context version
        output_file = steering_vectors_dir / f"steering_vector_layer{layer}_{args.n_shots}shot_{n_seeds}seeds_nocontext.pt"
        torch.save(result, output_file)

        # Summary for this layer
        print(f"\nLayer {layer} Summary:")
        print(f"  Total samples: {total_valid} ({n_seeds} seeds × ~429 samples)")
        print(f"  Total skipped: {total_skipped}")
        print(f"  Avg 0-shot length: {avg_len_0shot:.1f} tokens")
        print(f"  Avg {args.n_shots}-shot length: {avg_len_nshot:.1f} tokens")
        print(f"  PC1 variance explained: {var_explained_pc1:.2f}%")
        print(f"  Saved to: {output_file}")

    # Clean up temp worker files
    print("\nCleaning up temporary files...")
    for filepath in worker_files.values():
        try:
            Path(filepath).unlink()
        except:
            pass
    try:
        temp_dir.rmdir()
    except:
        pass

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Layers extracted: {args.layers}")
    print(f"Seeds used: {n_seeds} ({args.seeds[0]}-{args.seeds[-1]})")
    print(f"Token positions: last {args.n_tokens} tokens")
    print(f"Total representations: {total_valid} per layer")
    print(f"GPUs used: {n_gpus} (true data parallelism)")
    print(f"Files saved to: {steering_vectors_dir}")


if __name__ == "__main__":
    main()
