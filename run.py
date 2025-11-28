"""
Main entry point for FinQA evaluation.

Following the original FinQA benchmark:
https://github.com/czyssrs/FinQA

Supports two modes:
1. Direct Answer: Model outputs numerical answer directly
2. Program Synthesis: Model outputs DSL program, which is executed to get answer

Metrics (following original):
- Execution Accuracy: Whether computed answer matches gold (5 decimal places)
- Program Accuracy: Whether predicted program matches gold program
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from src.datasets import FinQADataset
from src.evaluate import (
    evaluate_predictions,
    evaluate_by_operation,
    evaluate_by_num_steps,
    execution_accuracy,
    program_accuracy,
    evaluate_program_execution,
    extract_answer_from_cot,
)
from src.model import VLLMInference
from src.program import execute_program, extract_program_from_text
from src.prompts import VanillaPrompt, FewShotPrompt, ChainOfThoughtPrompt

logger = logging.getLogger(__name__)

PROMPT_TYPES = {
    "vanilla": VanillaPrompt,
    "few_shot": FewShotPrompt,
    "cot": ChainOfThoughtPrompt,
}


def setup_logging(output_dir: Path) -> None:
    """Setup logging to both console and file."""
    log_file = output_dir / "run.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_session_name(
    model: str,
    prompt_type: str,
    n_shots: int,
    output_program: bool,
    tag: str | None = None,
) -> str:
    """Generate session folder name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("/")[-1]

    if prompt_type == "few_shot":
        mode = f"{n_shots}shot"
    else:
        mode = prompt_type

    # Add output mode indicator
    output_mode = "prog" if output_program else "ans"

    parts = [model_short, mode, output_mode, timestamp]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)


def run_evaluation(
    model: VLLMInference,
    dataset: FinQADataset,
    prompt_template,
    icl_pool: FinQADataset | None = None,
    n_shots: int = 0,
    max_samples: int | None = None,
    batch_size: int = 8,
    max_tokens: int = 256,
    output_program: bool = False,
    prompt_type: str = "vanilla",
    icl_seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    Run evaluation on the dataset.

    Args:
        model: vLLM model wrapper
        dataset: FinQA dataset (test set)
        prompt_template: Prompt template instance
        icl_pool: Dataset to sample ICL examples from (train set)
        n_shots: Number of ICL examples (for few-shot)
        max_samples: Maximum samples to evaluate (None for all)
        batch_size: Batch size for inference
        max_tokens: Maximum tokens to generate
        output_program: If True, expect program output and execute it
        icl_seed: Seed for ICL example sampling

    Returns:
        Tuple of (predictions list, metrics dict)
    """
    import random
    icl_rng = random.Random(icl_seed)
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    logger.info(f"Running evaluation on {n_samples} samples")
    logger.info(f"Output mode: {'Program Synthesis' if output_program else 'Direct Answer'}")

    all_predictions = []
    all_prompts = []
    all_ground_truths = []
    all_programs = []
    all_tables = []

    # Prepare prompts
    for idx in tqdm(range(n_samples), desc="Preparing prompts"):
        example = dataset[idx]
        question = example["question"]
        context = example["context"]
        ground_truth = example["answer"]
        program = example.get("program", "")
        table = example.get("metadata", {}).get("table", None)

        # Get ICL examples if needed
        icl_examples = None
        if n_shots > 0 and icl_pool is not None:
            icl_examples = dataset.get_icl_examples(n_shots, idx, icl_pool=icl_pool, rng=icl_rng)

        # Format prompt
        prompt = prompt_template.format(
            question=question,
            context=context,
            icl_examples=icl_examples,
        )

        all_prompts.append(prompt)
        all_ground_truths.append(ground_truth)
        all_programs.append(program)
        all_tables.append(table)

        # Convert prompt to string for saving
        if isinstance(prompt, list):
            # Chat format - extract content
            prompt_str = "\n".join([f"[{m['role']}]: {m['content']}" for m in prompt])
        else:
            prompt_str = str(prompt)

        all_predictions.append({
            "id": example.get("id", f"test_{idx}"),
            "idx": idx,
            "question": question,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "ground_truth": ground_truth,
            "gold_program": program,
            "full_prompt": prompt_str,
        })

    # Batch inference with stats tracking
    logger.info("Running inference...")
    all_responses = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch = all_prompts[i : i + batch_size]
        stats = model.generate(batch, max_tokens=max_tokens, return_stats=True)
        all_responses.extend(stats.responses)
        total_input_tokens += stats.total_input_tokens
        total_output_tokens += stats.total_output_tokens
        total_latency += stats.latency_seconds

    # Process predictions
    pred_programs = []
    pred_answers = []

    for pred, response, table, gold_program in zip(
        all_predictions, all_responses, all_tables, all_programs
    ):
        pred["raw_output"] = response

        if output_program:
            # Extract program from response
            extracted_program = extract_program_from_text(response)
            if extracted_program is None:
                # Try using the response directly as program
                extracted_program = response.strip()

            pred["pred_program"] = extracted_program
            pred_programs.append(extracted_program)

            # Execute the program to get answer
            exe_result = execute_program(extracted_program, table)
            if exe_result is not None:
                pred["prediction"] = str(exe_result)
            else:
                pred["prediction"] = response  # Fallback to raw output
            pred_answers.append(pred["prediction"])

            # Compute program accuracy
            pred["program_accuracy"] = program_accuracy(extracted_program, gold_program)

        else:
            # Direct answer mode
            pred["raw_output"] = response

            # For CoT, extract answer from reasoning
            if prompt_type == "cot":
                extracted = extract_answer_from_cot(response)
                pred["prediction"] = extracted
            else:
                pred["prediction"] = response

            pred_answers.append(pred["prediction"])
            pred_programs.append("")

        # Compute execution accuracy
        pred["execution_accuracy"] = execution_accuracy(
            pred["prediction"], pred["ground_truth"]
        )

    # Compute aggregate metrics
    metrics = evaluate_predictions(
        pred_answers,
        all_ground_truths,
        programs=all_programs,
        pred_programs=pred_programs if output_program else None,
    )

    # Add per-operation breakdown
    op_metrics = evaluate_by_operation(pred_answers, all_ground_truths, all_programs)
    metrics["by_operation"] = op_metrics

    # Add per-step breakdown
    step_metrics = evaluate_by_num_steps(pred_answers, all_ground_truths, all_programs)
    metrics["by_num_steps"] = step_metrics

    # Add inference stats
    metrics["inference_stats"] = {
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_per_sample_seconds": round(total_latency / n_samples, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_input_tokens_per_sample": round(total_input_tokens / n_samples, 1),
        "avg_output_tokens_per_sample": round(total_output_tokens / n_samples, 1),
        "avg_total_tokens_per_sample": round((total_input_tokens + total_output_tokens) / n_samples, 1),
        "tokens_per_second": round(total_output_tokens / total_latency, 1) if total_latency > 0 else 0,
    }

    return all_predictions, metrics


def save_predictions(predictions: list[dict], output_path: str) -> None:
    """Save predictions to JSON file."""
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {output_path}")


def save_prompts_readable(predictions: list[dict], output_dir: Path) -> None:
    """
    Save prompts as readable text files.

    Creates a 'prompts/' directory with individual .txt files for each prompt,
    so users can read the actual formatted prompts with proper newlines.
    """
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    for pred in predictions:
        idx = pred.get("idx", pred.get("id", "unknown"))
        prompt_file = prompts_dir / f"prompt_{idx}.txt"

        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(f"=== ID: {pred.get('id', '')} ===\n\n")
            f.write(pred.get("full_prompt", ""))
            f.write(f"\n\n=== RAW OUTPUT ===\n\n")
            f.write(pred.get("raw_output", ""))
            f.write(f"\n\n=== EXTRACTED ANSWER ===\n")
            f.write(f"Prediction: {pred.get('prediction', '')}\n")
            f.write(f"Ground Truth: {pred.get('ground_truth', '')}\n")
            f.write(f"Correct: {pred.get('execution_accuracy', 0)}\n")

    # Also save a combined file with all prompts
    combined_file = output_dir / "all_prompts.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions):
            f.write(f"\n{'='*80}\n")
            f.write(f"Example {i+1} / {len(predictions)} - ID: {pred.get('id', '')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(pred.get("full_prompt", ""))
            f.write(f"\n\n--- RAW OUTPUT ---\n")
            f.write(pred.get("raw_output", ""))
            f.write(f"\n\n--- RESULT ---\n")
            f.write(f"Prediction: {pred.get('prediction', '')}\n")
            f.write(f"Ground Truth: {pred.get('ground_truth', '')}\n")
            f.write(f"Correct: {'✓' if pred.get('execution_accuracy', 0) else '✗'}\n")

    logger.info(f"Readable prompts saved to {prompts_dir}/ and {combined_file}")


def save_comparison_excel(predictions: list[dict], output_path: str) -> None:
    """
    Save comparison Excel file with raw outputs.

    Columns: id, question, gold_answer, predicted_answer, raw_output, correct (1/0)
    """
    import pandas as pd

    rows = []
    for pred in predictions:
        rows.append({
            "id": pred.get("id", ""),
            "question": pred.get("question", "")[:200],  # Truncate for readability
            "gold_answer": pred.get("ground_truth", ""),
            "predicted_answer": pred.get("prediction", ""),
            "raw_output": pred.get("raw_output", "")[:1000],  # Truncate long outputs
            "correct": int(pred.get("execution_accuracy", 0)),
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, engine="openpyxl")

    # Summary stats
    total = len(df)
    correct = df["correct"].sum()
    logger.info(f"Comparison Excel saved to {output_path}")
    logger.info(f"  Total: {total}, Correct: {correct}, Accuracy: {correct/total:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="FinQA Evaluation (following original benchmark)"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="vanilla",
        choices=["vanilla", "few_shot", "cot"],
        help="Prompt template type",
    )
    parser.add_argument(
        "--output_program",
        action="store_true",
        help="If set, prompt model to output DSL program instead of direct answer",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=3,
        help="Number of ICL examples (for few_shot/cot)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=None, help="Max context length for vLLM")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_runs", type=int, default=1,
        help="Number of runs with different ICL samples (only for n_shots > 0)",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Optional tag for session name"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--no_context_in_examples",
        action="store_true",
        help="If set, ICL examples contain only question+reasoning+answer (no context)",
    )
    parser.add_argument(
        "--table_only_in_examples",
        action="store_true",
        help="If set, ICL examples contain table+question+reasoning+answer (no text passages)",
    )
    parser.add_argument(
        "--no_context_in_query",
        action="store_true",
        help="If set, the main query contains only question (no context/table)",
    )
    parser.add_argument(
        "--use_qa_json",
        action="store_true",
        help="If set, load data from qa.json (merged file) instead of split-specific files",
    )
    args = parser.parse_args()

    # Load config if provided (config values override defaults)
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    # Create session directory
    session_name = get_session_name(
        args.model, args.prompt_type, args.n_shots, args.output_program, args.tag
    )
    output_dir = Path(args.output_dir) / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("FinQA Evaluation (czyssrs/FinQA)")
    logger.info("=" * 60)
    logger.info(f"Session: {session_name}")
    logger.info(f"Output Mode: {'Program Synthesis' if args.output_program else 'Direct Answer'}")
    logger.info(f"Configuration: {vars(args)}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Initialize prompt template
    prompt_class = PROMPT_TYPES[args.prompt_type]
    prompt_kwargs = {}

    if args.prompt_type == "few_shot":
        prompt_kwargs["n_shots"] = args.n_shots
        prompt_kwargs["output_program"] = args.output_program  # few_shot still supports this
    elif args.prompt_type == "cot":
        prompt_kwargs["n_shots"] = args.n_shots if args.n_shots > 0 else 2
        # Support no-context mode for ICL examples
        if getattr(args, 'no_context_in_examples', False):
            prompt_kwargs["include_context_in_examples"] = False
        # Support table-only mode for ICL examples
        if getattr(args, 'table_only_in_examples', False):
            prompt_kwargs["table_only_in_examples"] = True
        # Support no-context mode for query
        if getattr(args, 'no_context_in_query', False):
            prompt_kwargs["no_context_in_query"] = True
    # vanilla prompt has no extra params

    prompt_template = prompt_class(**prompt_kwargs)
    logger.info(f"Using prompt type: {args.prompt_type}")

    # Initialize datasets
    use_qa_json = getattr(args, 'use_qa_json', False)
    test_dataset = FinQADataset(split=args.split, seed=args.seed, use_qa_json=use_qa_json)
    test_dataset.load()
    logger.info(f"Loaded {args.split} dataset with {len(test_dataset)} examples" +
                (" (from qa.json)" if use_qa_json else ""))

    # Load ICL pool - use same split as test set (all examples available for ICL)
    icl_pool = None
    if args.prompt_type in ["few_shot", "cot"] and args.n_shots > 0:
        icl_pool = FinQADataset(split=args.split, seed=args.seed, use_qa_json=use_qa_json)
        icl_pool.load()
        logger.info(f"Loaded {args.split} dataset with {len(icl_pool)} examples for ICL" +
                    (" (from qa.json)" if use_qa_json else ""))

    # Initialize model
    logger.info(f"Loading model: {args.model}")
    model = VLLMInference(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
    )

    # Determine number of runs
    n_runs = getattr(args, 'n_runs', 1)
    if args.n_shots == 0:
        n_runs = 1  # No point in multiple runs for 0-shot

    # Run evaluation(s)
    logger.info("Starting evaluation...")
    all_run_metrics = []
    all_run_predictions = []

    for run_idx in range(n_runs):
        # Use different seed for each run's ICL sampling
        icl_seed = args.seed + run_idx

        if n_runs > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"RUN {run_idx + 1}/{n_runs} (ICL seed: {icl_seed})")
            logger.info(f"{'='*60}")

        predictions, metrics = run_evaluation(
            model=model,
            dataset=test_dataset,
            prompt_template=prompt_template,
            icl_pool=icl_pool,
            n_shots=args.n_shots,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            output_program=args.output_program,
            prompt_type=args.prompt_type,
            icl_seed=icl_seed,
        )

        metrics["run_idx"] = run_idx
        metrics["icl_seed"] = icl_seed
        all_run_metrics.append(metrics)
        all_run_predictions.append(predictions)

        if n_runs > 1:
            logger.info(f"Run {run_idx + 1} Accuracy: {metrics['execution_accuracy']:.4f}")

    # Aggregate results across runs
    if n_runs > 1:
        import numpy as np
        accuracies = [m['execution_accuracy'] for m in all_run_metrics]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)

        aggregate_metrics = {
            "n_runs": n_runs,
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "min_accuracy": float(min_acc),
            "max_accuracy": float(max_acc),
            "all_accuracies": accuracies,
            "per_run_metrics": all_run_metrics,
        }

        # Save aggregate metrics
        with open(output_dir / "aggregate_metrics.json", "w") as f:
            json.dump(aggregate_metrics, f, indent=2)

        # Save per-run predictions
        for run_idx, preds in enumerate(all_run_predictions):
            save_predictions(preds, str(output_dir / f"predictions_run{run_idx}.json"))

        # Save first run's readable prompts as example
        save_prompts_readable(all_run_predictions[0], output_dir)

        # Save comparison Excel for first run
        save_comparison_excel(all_run_predictions[0], str(output_dir / "comparison_run0.xlsx"))

        # Summary for multiple runs
        logger.info("\n" + "=" * 60)
        logger.info("AGGREGATE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Number of runs: {n_runs}")
        logger.info(f"Samples per run: {all_run_metrics[0]['n_samples']}")
        logger.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"Min Accuracy: {min_acc:.4f}")
        logger.info(f"Max Accuracy: {max_acc:.4f}")
        logger.info(f"All accuracies: {[f'{a:.4f}' for a in accuracies]}")

        metrics = aggregate_metrics  # Return aggregate
    else:
        # Single run - save as before
        save_predictions(predictions, str(output_dir / "predictions.json"))
        save_prompts_readable(predictions, output_dir)
        save_comparison_excel(predictions, str(output_dir / "comparison.xlsx"))

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Summary
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Samples evaluated: {metrics['n_samples']}")
        logger.info(
            f"Execution Accuracy: {metrics['execution_accuracy']:.4f} "
            f"({metrics['n_correct']}/{metrics['n_samples']})"
        )
        if "program_accuracy" in metrics:
            logger.info(f"Program Accuracy: {metrics['program_accuracy']:.4f}")

        logger.info("-" * 40)
        logger.info("Accuracy by Operation:")
        for op, op_metrics in metrics["by_operation"].items():
            if op_metrics["total"] > 0:
                logger.info(
                    f"  {op}: {op_metrics['accuracy']:.4f} ({op_metrics['total']} samples)"
                )

        logger.info("-" * 40)
        logger.info("Accuracy by Number of Steps:")
        for step_key in sorted(metrics["by_num_steps"].keys()):
            step_metrics = metrics["by_num_steps"][step_key]
            if step_metrics["total"] > 0:
                logger.info(
                    f"  {step_key}: {step_metrics['accuracy']:.4f} ({step_metrics['total']} samples)"
                )

        # Log inference stats
        if "inference_stats" in metrics:
            inf_stats = metrics["inference_stats"]
            logger.info("-" * 40)
            logger.info("Inference Statistics:")
            logger.info(f"  Total latency: {inf_stats['total_latency_seconds']:.2f}s")
            logger.info(f"  Avg latency per sample: {inf_stats['avg_latency_per_sample_seconds']:.4f}s")
            logger.info(f"  Total tokens: {inf_stats['total_tokens']:,}")
            logger.info(f"  Avg input tokens: {inf_stats['avg_input_tokens_per_sample']:.1f}")
            logger.info(f"  Avg output tokens: {inf_stats['avg_output_tokens_per_sample']:.1f}")
            logger.info(f"  Tokens/second: {inf_stats['tokens_per_second']:.1f}")

    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
