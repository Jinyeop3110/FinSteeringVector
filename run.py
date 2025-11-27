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

    Returns:
        Tuple of (predictions list, metrics dict)
    """
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
            icl_examples = dataset.get_icl_examples(n_shots, idx, icl_pool=icl_pool)

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
        all_predictions.append({
            "id": example.get("id", f"test_{idx}"),
            "idx": idx,
            "question": question,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "ground_truth": ground_truth,
            "gold_program": program,
        })

    # Batch inference
    logger.info("Running inference...")
    all_responses = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch = all_prompts[i : i + batch_size]
        responses = model.generate(batch, max_tokens=max_tokens)
        all_responses.extend(responses)

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
            pred["prediction"] = response
            pred_answers.append(response)
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

    return all_predictions, metrics


def save_predictions(predictions: list[dict], output_path: str) -> None:
    """Save predictions to JSON file."""
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {output_path}")


def save_comparison_excel(predictions: list[dict], output_path: str) -> None:
    """
    Save minimal comparison Excel file.

    Columns: id, question, gold_answer, predicted_answer, correct (1/0)
    """
    import pandas as pd

    rows = []
    for pred in predictions:
        rows.append({
            "id": pred.get("id", ""),
            "question": pred.get("question", "")[:100],  # Truncate for readability
            "gold_answer": pred.get("ground_truth", ""),
            "predicted_answer": pred.get("prediction", ""),
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
        "--tag", type=str, default=None, help="Optional tag for session name"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate"
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
    prompt_kwargs = {"output_program": args.output_program}

    if args.prompt_type == "few_shot":
        prompt_kwargs["n_shots"] = args.n_shots
    elif args.prompt_type == "cot":
        prompt_kwargs["n_shots"] = args.n_shots if args.n_shots > 0 else 2

    prompt_template = prompt_class(**prompt_kwargs)
    logger.info(f"Using prompt type: {args.prompt_type}")

    # Initialize datasets
    test_dataset = FinQADataset(split=args.split, seed=args.seed)
    test_dataset.load()
    logger.info(f"Loaded {args.split} dataset with {len(test_dataset)} examples")

    # Load train set for ICL examples if needed
    icl_pool = None
    if args.prompt_type in ["few_shot", "cot"] and args.n_shots > 0:
        icl_pool = FinQADataset(split="train", seed=args.seed)
        icl_pool.load()
        logger.info(f"Loaded train dataset with {len(icl_pool)} examples for ICL")

    # Initialize model
    logger.info(f"Loading model: {args.model}")
    model = VLLMInference(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
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
    )

    # Save results
    logger.info("Saving results...")
    save_predictions(predictions, str(output_dir / "predictions.json"))

    # Comparison Excel
    save_comparison_excel(predictions, str(output_dir / "comparison.xlsx"))

    # Metrics JSON
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

    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
