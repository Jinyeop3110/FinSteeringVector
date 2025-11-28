#!/usr/bin/env python3
"""Generate accuracy and latency plots from experiment results."""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(outputs_dir: Path, experiment_date: str = None) -> dict:
    """
    Load results from experiment directories.

    Args:
        outputs_dir: Path to outputs directory
        experiment_date: Optional date filter (YYYYMMDD format)

    Returns:
        Dictionary with experiment results
    """
    results = {
        "vanilla": [],
        "cot": []
    }

    for exp_dir in sorted(outputs_dir.glob("*")):
        if not exp_dir.is_dir():
            continue

        name = exp_dir.name

        # Filter by date if specified
        if experiment_date and experiment_date not in name:
            continue

        # Parse experiment type and shots from directory name
        if "vanilla" in name.lower():
            exp_type = "vanilla"
            n_shots = 0
        elif "cot" in name.lower():
            exp_type = "cot"
            # Extract n_shots from name like "cot_5shot_nocontext_..."
            import re
            match = re.search(r'(\d+)shot', name)
            n_shots = int(match.group(1)) if match else 0
        else:
            continue

        # Load metrics
        agg_metrics_file = exp_dir / "aggregate_metrics.json"
        metrics_file = exp_dir / "metrics.json"

        if agg_metrics_file.exists():
            with open(agg_metrics_file) as f:
                metrics = json.load(f)
            accuracy = metrics.get("mean_accuracy", 0)
            std = metrics.get("std_accuracy", 0)
            n_runs = metrics.get("n_runs", 1)
            # Get inference stats from first run
            if "per_run_metrics" in metrics and metrics["per_run_metrics"]:
                inf_stats = metrics["per_run_metrics"][0].get("inference_stats", {})
            else:
                inf_stats = {}
        elif metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            accuracy = metrics.get("execution_accuracy", 0)
            std = 0
            n_runs = 1
            inf_stats = metrics.get("inference_stats", {})
        else:
            continue

        result = {
            "name": name,
            "n_shots": n_shots,
            "accuracy": accuracy,
            "std": std,
            "n_runs": n_runs,
            "latency": inf_stats.get("total_latency_seconds", 0),
            "avg_latency": inf_stats.get("avg_latency_per_sample_seconds", 0),
            "input_tokens": inf_stats.get("avg_input_tokens_per_sample", 0),
            "output_tokens": inf_stats.get("avg_output_tokens_per_sample", 0),
            "total_tokens": inf_stats.get("total_tokens", 0),
        }

        results[exp_type].append(result)

    # Sort by n_shots and filter to 0-4 shots only
    for exp_type in results:
        results[exp_type] = [r for r in results[exp_type] if r["n_shots"] <= 4]
        results[exp_type].sort(key=lambda x: x["n_shots"])

    return results


def plot_accuracy_vs_shots(results: dict, output_path: Path):
    """Plot accuracy vs number of shots with error bars."""
    fig, ax = plt.subplots(figsize=(6.7, 4))

    # CoT results
    cot_results = results["cot"]
    if cot_results:
        shots = [r["n_shots"] for r in cot_results]
        accs = [r["accuracy"] * 100 for r in cot_results]
        stds = [r["std"] * 100 for r in cot_results]

        ax.errorbar(shots, accs, yerr=stds, marker='o', capsize=5,
                   label='CoT', linewidth=2, markersize=8, color='#2E86AB')
        ax.fill_between(shots,
                       [a - s for a, s in zip(accs, stds)],
                       [a + s for a, s in zip(accs, stds)],
                       alpha=0.2, color='#2E86AB')

    # Vanilla baseline (horizontal line)
    vanilla_results = results["vanilla"]
    if vanilla_results:
        vanilla_acc = vanilla_results[0]["accuracy"] * 100
        vanilla_std = vanilla_results[0]["std"] * 100
        ax.axhline(y=vanilla_acc, color='#E94F37', linestyle='--',
                  linewidth=2, label=f'Vanilla 0-shot ({vanilla_acc:.1f}%)')
        if vanilla_std > 0:
            ax.axhspan(vanilla_acc - vanilla_std, vanilla_acc + vanilla_std,
                      alpha=0.1, color='#E94F37')

    ax.set_xlabel('Number of ICL Examples (shots)', fontsize=12)
    ax.set_ylabel('Execution Accuracy (%)', fontsize=12)
    ax.set_title('FinQA Accuracy vs ICL Examples\n(Qwen2.5-1.5B-Instruct, 256 samples, 4 seeds)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 5))
    ax.set_xlim(-0.5, 4.5)

    # Set y-axis limits with some padding
    if cot_results:
        all_accs = [r["accuracy"] * 100 for r in cot_results]
        if vanilla_results:
            all_accs.append(vanilla_results[0]["accuracy"] * 100)
        y_min = max(0, min(all_accs) - 10)
        y_max = min(100, max(all_accs) + 10)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to: {output_path}")


def plot_latency_comparison(results: dict, output_path: Path):
    """Plot latency and token usage comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.3))

    # Combine all results
    all_results = []
    if results["vanilla"]:
        all_results.extend([(r, "Vanilla") for r in results["vanilla"]])
    if results["cot"]:
        all_results.extend([(r, "CoT") for r in results["cot"]])

    all_results.sort(key=lambda x: (0 if x[1] == "Vanilla" else 1, x[0]["n_shots"]))

    labels = []
    latencies = []
    input_tokens = []
    output_tokens = []
    colors = []

    for r, exp_type in all_results:
        if exp_type == "Vanilla":
            labels.append("Vanilla\n0-shot")
            colors.append('#E94F37')
        else:
            labels.append(f"CoT\n{r['n_shots']}-shot")
            colors.append('#2E86AB')
        latencies.append(r["latency"])
        input_tokens.append(r["input_tokens"])
        output_tokens.append(r["output_tokens"])

    x = np.arange(len(labels))

    # Left plot: Total latency
    ax1 = axes[0]
    bars1 = ax1.bar(x, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Total Latency (seconds)', fontsize=11)
    ax1.set_title('Inference Latency (256 samples)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, lat in zip(bars1, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{lat:.1f}s', ha='center', va='bottom', fontsize=8)

    # Right plot: Token usage (stacked bar)
    ax2 = axes[1]
    width = 0.6
    bars_in = ax2.bar(x, input_tokens, width, label='Input Tokens', color='#A8DADC', edgecolor='black', linewidth=0.5)
    bars_out = ax2.bar(x, output_tokens, width, bottom=input_tokens, label='Output Tokens', color='#457B9D', edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Avg Tokens per Sample', fontsize=11)
    ax2.set_title('Token Usage per Sample', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (inp, out) in enumerate(zip(input_tokens, output_tokens)):
        total = inp + out
        ax2.text(i, total + 50, f'{total:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latency plot to: {output_path}")


def plot_accuracy_bar_simple(results: dict, output_path: Path):
    """Plot simple bar chart comparing Vanilla, CoT 0-shot, and CoT 3-shot."""
    fig, ax = plt.subplots(figsize=(5.3, 4))

    # Collect data for the three conditions
    labels = []
    accuracies = []
    stds = []
    colors = []

    # Vanilla 0-shot
    vanilla_results = results["vanilla"]
    if vanilla_results:
        labels.append("Vanilla\n0-shot")
        accuracies.append(vanilla_results[0]["accuracy"] * 100)
        stds.append(vanilla_results[0]["std"] * 100)
        colors.append('#E94F37')

    # CoT 0-shot and 3-shot
    cot_results = results["cot"]
    for r in cot_results:
        if r["n_shots"] == 0:
            labels.append("CoT\n0-shot")
            accuracies.append(r["accuracy"] * 100)
            stds.append(r["std"] * 100)
            colors.append('#2E86AB')
        elif r["n_shots"] == 3:
            labels.append("CoT\n3-shot")
            accuracies.append(r["accuracy"] * 100)
            stds.append(r["std"] * 100)
            colors.append('#1B4965')

    if not labels:
        print("No data found for Vanilla, CoT 0-shot, or CoT 3-shot")
        return

    x = np.arange(len(labels))
    width = 0.6

    bars = ax.bar(x, accuracies, width, yerr=stds, capsize=8,
                  color=colors, edgecolor='black', linewidth=1, alpha=0.85)

    # Add value labels on bars
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Execution Accuracy (%)', fontsize=12)
    ax.set_title('FinQA Accuracy: Vanilla vs Chain-of-Thought\n(Qwen2.5-1.5B-Instruct)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(accuracies) + max(stds) + 10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add a baseline at 0
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved simple bar plot to: {output_path}")


def plot_combined(results: dict, output_path: Path):
    """Create a combined figure with accuracy, latency, and efficiency."""
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.7))

    # Plot 1: Accuracy vs Shots (top-left)
    ax1 = axes[0, 0]
    cot_results = results["cot"]
    if cot_results:
        shots = [r["n_shots"] for r in cot_results]
        accs = [r["accuracy"] * 100 for r in cot_results]
        stds = [r["std"] * 100 for r in cot_results]

        ax1.errorbar(shots, accs, yerr=stds, marker='o', capsize=5,
                    linewidth=2, markersize=8, color='#2E86AB', label='CoT')
        ax1.fill_between(shots,
                        [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)],
                        alpha=0.2, color='#2E86AB')

    vanilla_results = results["vanilla"]
    if vanilla_results:
        vanilla_acc = vanilla_results[0]["accuracy"] * 100
        ax1.axhline(y=vanilla_acc, color='#E94F37', linestyle='--',
                   linewidth=2, label=f'Vanilla ({vanilla_acc:.1f}%)')

    ax1.set_xlabel('Number of ICL Examples', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('(a) Accuracy vs ICL Examples', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 5))

    # Plot 2: Latency vs Shots (top-right)
    ax2 = axes[0, 1]
    if cot_results:
        shots = [r["n_shots"] for r in cot_results]
        latencies = [r["latency"] for r in cot_results]
        ax2.plot(shots, latencies, marker='s', linewidth=2, markersize=8, color='#2E86AB', label='CoT')

    if vanilla_results:
        vanilla_lat = vanilla_results[0]["latency"]
        ax2.axhline(y=vanilla_lat, color='#E94F37', linestyle='--',
                   linewidth=2, label=f'Vanilla ({vanilla_lat:.1f}s)')

    ax2.set_xlabel('Number of ICL Examples', fontsize=11)
    ax2.set_ylabel('Total Latency (s)', fontsize=11)
    ax2.set_title('(b) Latency vs ICL Examples', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 5))

    # Plot 3: Input Tokens vs Shots (bottom-left)
    ax3 = axes[1, 0]
    if cot_results:
        shots = [r["n_shots"] for r in cot_results]
        input_toks = [r["input_tokens"] for r in cot_results]
        output_toks = [r["output_tokens"] for r in cot_results]

        ax3.plot(shots, input_toks, marker='^', linewidth=2, markersize=8,
                color='#A8DADC', label='Input Tokens')
        ax3.plot(shots, output_toks, marker='v', linewidth=2, markersize=8,
                color='#457B9D', label='Output Tokens')

    if vanilla_results:
        ax3.axhline(y=vanilla_results[0]["input_tokens"], color='#E94F37',
                   linestyle='--', linewidth=1, alpha=0.7)

    ax3.set_xlabel('Number of ICL Examples', fontsize=11)
    ax3.set_ylabel('Avg Tokens per Sample', fontsize=11)
    ax3.set_title('(c) Token Usage vs ICL Examples', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 5))

    # Plot 4: Efficiency (Accuracy per second) (bottom-right)
    ax4 = axes[1, 1]
    if cot_results:
        shots = [r["n_shots"] for r in cot_results]
        efficiency = [r["accuracy"] * 100 / r["latency"] if r["latency"] > 0 else 0
                     for r in cot_results]
        ax4.plot(shots, efficiency, marker='D', linewidth=2, markersize=8,
                color='#2E86AB', label='CoT')

    if vanilla_results:
        vanilla_eff = vanilla_results[0]["accuracy"] * 100 / vanilla_results[0]["latency"]
        ax4.axhline(y=vanilla_eff, color='#E94F37', linestyle='--',
                   linewidth=2, label=f'Vanilla ({vanilla_eff:.2f})')

    ax4.set_xlabel('Number of ICL Examples', fontsize=11)
    ax4.set_ylabel('Accuracy % / Second', fontsize=11)
    ax4.set_title('(d) Efficiency (Accuracy/Latency)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(0, 5))

    plt.suptitle('FinQA Performance Analysis: Qwen2.5-1.5B-Instruct\n(256 samples, 4 seeds per config)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to: {output_path}")


def generate_results_table(results: dict, output_path: Path):
    """Generate markdown table of results."""
    lines = [
        "# FinQA Experiment Results",
        "",
        f"**Model:** Qwen/Qwen2.5-1.5B-Instruct",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** 256",
        f"**Seeds:** 4 per configuration",
        "",
        "## Results Table",
        "",
        "| Config | Shots | Accuracy | Std | Latency (s) | Input Tokens | Output Tokens |",
        "|--------|-------|----------|-----|-------------|--------------|---------------|",
    ]

    # Add vanilla
    for r in results["vanilla"]:
        lines.append(
            f"| Vanilla | {r['n_shots']} | {r['accuracy']*100:.2f}% | "
            f"{r['std']*100:.2f}% | {r['latency']:.1f} | "
            f"{r['input_tokens']:.0f} | {r['output_tokens']:.0f} |"
        )

    # Add CoT
    for r in results["cot"]:
        lines.append(
            f"| CoT | {r['n_shots']} | {r['accuracy']*100:.2f}% | "
            f"{r['std']*100:.2f}% | {r['latency']:.1f} | "
            f"{r['input_tokens']:.0f} | {r['output_tokens']:.0f} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **CoT vs Vanilla:** Chain-of-thought prompting significantly improves accuracy",
        "2. **ICL Scaling:** Accuracy generally improves with more examples up to a saturation point",
        "3. **Latency Trade-off:** More examples increase input tokens but latency is dominated by output generation",
        "",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved results table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from FinQA experiment results")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Path to outputs directory")
    parser.add_argument("--report_dir", type=str, default="figures/1_evaluation", help="Path to save figures")
    parser.add_argument("--date", type=str, default=None, help="Filter by date (YYYYMMDD)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading experiment results...")
    results = load_experiment_results(outputs_dir, args.date)

    n_vanilla = len(results["vanilla"])
    n_cot = len(results["cot"])
    print(f"Found {n_vanilla} vanilla and {n_cot} CoT experiments")

    if n_vanilla == 0 and n_cot == 0:
        print("No experiments found! Run experiments first.")
        return

    # Generate plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_accuracy_vs_shots(results, report_dir / f"accuracy_vs_shots_{timestamp}.png")
    plot_accuracy_bar_simple(results, report_dir / f"accuracy_bar_simple_{timestamp}.png")
    plot_latency_comparison(results, report_dir / f"latency_comparison_{timestamp}.png")
    plot_combined(results, report_dir / f"combined_analysis_{timestamp}.png")
    generate_results_table(results, report_dir / f"results_{timestamp}.md")

    print("\nDone! Reports saved to:", report_dir)


if __name__ == "__main__":
    main()
