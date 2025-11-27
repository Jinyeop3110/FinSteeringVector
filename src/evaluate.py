"""
Evaluation metrics for FinQA (Financial Question Answering).

Following the original FinQA evaluation:
https://github.com/czyssrs/FinQA

Two main metrics:
1. Execution Accuracy: Whether the executed result matches the gold answer
2. Program Accuracy: Whether the predicted program matches the gold program
"""

import re
from typing import Optional

from .program import execute_program, str_to_num, parse_program


def extract_number(text: str) -> Optional[float]:
    """
    Extract numerical value from text.

    Handles:
    - Plain numbers: 123, 123.45
    - Percentages: 12.5%, 12.5 %
    - Currency: $1,234.56
    - Negative: -123, (123)
    - Scientific: 1.23e-4
    - Yes/No: yes -> 1.0, no -> 0.0

    Args:
        text: Text containing a number

    Returns:
        Extracted float or None if no number found
    """
    if not text:
        return None

    text = str(text).strip()

    # Handle yes/no for greater operation results
    if text.lower() == "yes":
        return 1.0
    if text.lower() == "no":
        return 0.0

    # Try to find the last number in the text (usually the answer)
    patterns = [
        r"[-+]?\d+\.?\d*%",  # Percentage
        r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?",  # Currency with commas
        r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?",  # Scientific notation
        r"\([\d,.]+\)",  # Negative in parentheses
    ]

    combined = "|".join(f"({p})" for p in patterns)
    matches = re.findall(combined, text)

    if not matches:
        return None

    # Get the last match (flatten tuple and filter empty)
    last_match = None
    for match_tuple in matches:
        for m in match_tuple:
            if m:
                last_match = m

    if not last_match:
        return None

    # Clean and convert
    num_str = last_match.strip()

    # Handle percentage
    is_percent = "%" in num_str
    num_str = num_str.replace("%", "")

    # Handle currency and commas
    num_str = num_str.replace("$", "").replace(",", "")

    # Handle parentheses for negative
    if num_str.startswith("(") and num_str.endswith(")"):
        num_str = "-" + num_str[1:-1]

    try:
        value = float(num_str)
        if is_percent:
            value = value / 100
        return value
    except ValueError:
        return None


def execution_accuracy(
    prediction: str,
    ground_truth: str,
) -> float:
    """
    FinQA execution accuracy metric (following original implementation).

    Compares values after rounding to 5 decimal places.
    Also handles percentage/decimal format mismatches:
    - If model outputs 14.6 (meaning 14.6%) and gold is 0.146, they should match

    Args:
        prediction: Model prediction (number or text containing number)
        ground_truth: Ground truth (exe_ans from dataset)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Handle yes/no comparison first
    pred_lower = str(prediction).strip().lower()
    gt_lower = str(ground_truth).strip().lower()

    # Check for yes/no in prediction text
    if "yes" in pred_lower or "no" in pred_lower:
        if gt_lower in ["yes", "no"]:
            # Extract yes/no from prediction
            if "yes" in pred_lower and "no" not in pred_lower:
                return float(gt_lower == "yes")
            elif "no" in pred_lower and "yes" not in pred_lower:
                return float(gt_lower == "no")

    if gt_lower in ["yes", "no"]:
        return 0.0  # Gold is yes/no but prediction doesn't contain it

    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is None or gt_num is None:
        return 0.0

    # Round to 5 decimal places (following original FinQA)
    pred_rounded = round(pred_num, 5)
    gt_rounded = round(gt_num, 5)

    # Direct match
    if pred_rounded == gt_rounded:
        return 1.0

    # Handle percentage/decimal format mismatch
    # Case 1: Model outputs percentage form (e.g., 14.6) but gold is decimal (0.146)
    # This happens when model outputs "14.6" meaning 14.6% without the % sign
    if gt_num != 0 and 0 < abs(gt_num) < 1:  # Gold looks like a decimal percentage
        pred_as_decimal = round(pred_num / 100, 5)
        if pred_as_decimal == gt_rounded:
            return 1.0

    # Case 2: Model outputs decimal (0.146) but gold is percentage form (14.6)
    # Less common but handle for completeness
    if pred_num != 0 and 0 < abs(pred_num) < 1:
        pred_as_percent = round(pred_num * 100, 5)
        if pred_as_percent == gt_rounded:
            return 1.0

    return 0.0


def program_accuracy(
    pred_program: str,
    gold_program: str,
) -> float:
    """
    Program accuracy metric.

    Compares predicted program tokens against gold program tokens.

    Args:
        pred_program: Predicted program string
        gold_program: Gold program string

    Returns:
        1.0 if programs match, 0.0 otherwise
    """
    if not pred_program or not gold_program:
        return 0.0

    # Parse both programs
    pred_ops = parse_program(pred_program)
    gold_ops = parse_program(gold_program)

    if len(pred_ops) != len(gold_ops):
        return 0.0

    # Compare each operation
    for (p_op, p_a1, p_a2), (g_op, g_a1, g_a2) in zip(pred_ops, gold_ops):
        if p_op != g_op:
            return 0.0

        # Normalize arguments for comparison
        def normalize_arg(arg):
            arg = str(arg).strip().lower()
            # Try to convert to number for comparison
            num = str_to_num(arg)
            if num is not None:
                return round(num, 5)
            return arg

        if normalize_arg(p_a1) != normalize_arg(g_a1):
            return 0.0
        if normalize_arg(p_a2) != normalize_arg(g_a2):
            return 0.0

    return 1.0


def evaluate_program_execution(
    pred_program: str,
    gold_answer: str,
    table: Optional[list] = None,
) -> float:
    """
    Evaluate by executing the predicted program.

    Args:
        pred_program: Predicted program string
        gold_answer: Gold answer (exe_ans)
        table: Optional table data for table operations

    Returns:
        1.0 if execution result matches gold answer, 0.0 otherwise
    """
    if not pred_program:
        return 0.0

    # Execute the predicted program
    result = execute_program(pred_program, table)

    if result is None:
        return 0.0

    # Compare with gold answer
    gold_num = extract_number(gold_answer)

    if gold_num is None:
        return 0.0

    # Round both to 5 decimal places
    result_rounded = round(result, 5)
    gold_rounded = round(gold_num, 5)

    return float(result_rounded == gold_rounded)


def evaluate_predictions(
    predictions: list[str],
    ground_truths: list[str],
    programs: Optional[list[str]] = None,
    pred_programs: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Evaluate predictions using FinQA metrics.

    Args:
        predictions: List of model predictions (answers)
        ground_truths: List of ground truth answers
        programs: Optional list of gold programs
        pred_programs: Optional list of predicted programs

    Returns:
        Dictionary with metric scores
    """
    assert len(predictions) == len(ground_truths), "Length mismatch"

    exec_acc_scores = []
    prog_acc_scores = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        exec_acc_scores.append(execution_accuracy(pred, gt))

        # Program accuracy if both programs provided
        if pred_programs and programs and i < len(pred_programs) and i < len(programs):
            prog_acc_scores.append(program_accuracy(pred_programs[i], programs[i]))

    results = {
        "execution_accuracy": sum(exec_acc_scores) / len(exec_acc_scores),
        "n_samples": len(predictions),
        "n_correct": int(sum(exec_acc_scores)),
    }

    if prog_acc_scores:
        results["program_accuracy"] = sum(prog_acc_scores) / len(prog_acc_scores)

    return results


def evaluate_by_operation(
    predictions: list[str],
    ground_truths: list[str],
    programs: list[str],
) -> dict[str, dict]:
    """
    Evaluate predictions grouped by operation type.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        programs: List of program strings (to extract operation)

    Returns:
        Dictionary with per-operation metrics
    """
    operations = [
        "add", "subtract", "multiply", "divide",
        "exp", "greater", "table_sum", "table_average",
        "table_max", "table_min",
    ]

    results = {op: {"correct": 0, "total": 0} for op in operations}
    results["multi_step"] = {"correct": 0, "total": 0}

    for pred, gt, prog in zip(predictions, ground_truths, programs):
        score = execution_accuracy(pred, gt)

        # Count number of operations
        ops_in_prog = parse_program(prog)
        is_multi_step = len(ops_in_prog) > 1

        if is_multi_step:
            results["multi_step"]["correct"] += score
            results["multi_step"]["total"] += 1

        # Categorize by first operation
        found_op = False
        for op in operations:
            if op in prog:
                results[op]["correct"] += score
                results[op]["total"] += 1
                found_op = True
                break

    # Compute accuracy per operation
    for op in results:
        total = results[op]["total"]
        if total > 0:
            results[op]["accuracy"] = results[op]["correct"] / total
        else:
            results[op]["accuracy"] = 0.0

    return results


def evaluate_by_num_steps(
    predictions: list[str],
    ground_truths: list[str],
    programs: list[str],
) -> dict[str, dict]:
    """
    Evaluate predictions grouped by number of reasoning steps.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        programs: List of program strings

    Returns:
        Dictionary with per-step-count metrics
    """
    results = {}

    for pred, gt, prog in zip(predictions, ground_truths, programs):
        score = execution_accuracy(pred, gt)

        # Count steps
        ops = parse_program(prog)
        n_steps = len(ops)
        key = f"{n_steps}_step" if n_steps == 1 else f"{n_steps}_steps"

        if key not in results:
            results[key] = {"correct": 0, "total": 0}

        results[key]["correct"] += score
        results[key]["total"] += 1

    # Compute accuracy per step count
    for key in results:
        total = results[key]["total"]
        if total > 0:
            results[key]["accuracy"] = results[key]["correct"] / total
        else:
            results[key]["accuracy"] = 0.0

    return results
