#!/usr/bin/env python3
"""Generate FSV (Financial Steering Vector) plots from experiment results."""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import re


def load_fsv_results(outputs_dir: Path) -> dict:
    """
    Load FSV results from outputs_SV directory.

    Returns:
        Dictionary with structure: {layer: {scale: accuracy}}
    """
    results = {}

    for exp_dir in sorted(outputs_dir.glob("sv_*")):
        if not exp_dir.is_dir():
            continue

        name = exp_dir.name

        # Parse layer from directory name
        layer_match = re.search(r'layer(\d+)', name)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))

        # Parse scale from directory name (e.g., scale0_2 -> 0.2)
        scale_match = re.search(r'scale(\d+)_(\d+)', name)
        if not scale_match:
            continue
        scale = float(f"{scale_match.group(1)}.{scale_match.group(2)}")

        # Load metrics
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        accuracy = metrics.get("execution_accuracy", 0) * 100
        inference_stats = metrics.get("inference_stats", {})

        if layer not in results:
            results[layer] = {}

        results[layer][scale] = {
            "accuracy": accuracy,
            "latency": inference_stats.get("total_latency_seconds", 0),
            "avg_latency": inference_stats.get("avg_latency_per_sample_seconds", 0),
            "input_tokens": inference_stats.get("avg_input_tokens_per_sample", 0),
            "output_tokens": inference_stats.get("avg_output_tokens_per_sample", 0),
        }

    return results


def load_baseline_results(outputs_dir: Path) -> dict:
    """Load baseline (0-shot CoT and 3-shot CoT) results."""
    baselines = {}

    # Look for CoT 0-shot results
    for exp_dir in sorted(outputs_dir.glob("cot_0shot*")):
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / "metrics.json"
        agg_metrics_file = exp_dir / "aggregate_metrics.json"

        if agg_metrics_file.exists():
            with open(agg_metrics_file) as f:
                metrics = json.load(f)
            baselines["cot_0shot"] = {
                "accuracy": metrics.get("mean_accuracy", 0) * 100,
                "std": metrics.get("std_accuracy", 0) * 100,
            }
        elif metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            baselines["cot_0shot"] = {
                "accuracy": metrics.get("execution_accuracy", 0) * 100,
                "std": 0,
                "input_tokens": metrics.get("inference_stats", {}).get("avg_input_tokens_per_sample", 0),
                "latency": metrics.get("inference_stats", {}).get("total_latency_seconds", 0),
            }
        break

    # Look for CoT 3-shot results
    for exp_dir in sorted(outputs_dir.glob("cot_3shot*")):
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / "metrics.json"
        agg_metrics_file = exp_dir / "aggregate_metrics.json"

        if agg_metrics_file.exists():
            with open(agg_metrics_file) as f:
                metrics = json.load(f)
            baselines["cot_3shot"] = {
                "accuracy": metrics.get("mean_accuracy", 0) * 100,
                "std": metrics.get("std_accuracy", 0) * 100,
            }
            # Get inference stats from first run
            if "per_run_metrics" in metrics and metrics["per_run_metrics"]:
                inf_stats = metrics["per_run_metrics"][0].get("inference_stats", {})
                baselines["cot_3shot"]["input_tokens"] = inf_stats.get("avg_input_tokens_per_sample", 0)
                baselines["cot_3shot"]["latency"] = inf_stats.get("total_latency_seconds", 0)
        elif metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            baselines["cot_3shot"] = {
                "accuracy": metrics.get("execution_accuracy", 0) * 100,
                "std": 0,
                "input_tokens": metrics.get("inference_stats", {}).get("avg_input_tokens_per_sample", 0),
                "latency": metrics.get("inference_stats", {}).get("total_latency_seconds", 0),
            }
        break

    return baselines


def plot_accuracy_vs_scale(fsv_results: dict, baselines: dict, output_path: Path):
    """Plot accuracy vs scaling factor (alpha) for different layers."""
    fig, ax = plt.subplots(figsize=(4.5, 2.7))

    colors = {12: '#E94F37', 16: '#2E86AB'}
    markers = {12: 'o', 16: 's'}

    # Plot FSV results for each layer
    for layer in sorted(fsv_results.keys()):
        if layer not in [12, 16]:  # Only plot layers 12 and 16
            continue

        scales = sorted(fsv_results[layer].keys())
        accs = [fsv_results[layer][s]["accuracy"] for s in scales]

        ax.plot(scales, accs, marker=markers[layer], linewidth=1.5, markersize=5,
                color=colors[layer], label=f'Layer {layer}')

    # Add baseline horizontal lines
    if "cot_0shot" in baselines:
        baseline_acc = baselines["cot_0shot"]["accuracy"]
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5,
                   label=f'CoT 0-shot ({baseline_acc:.1f}%)')

    if "cot_3shot" in baselines:
        target_acc = baselines["cot_3shot"]["accuracy"]
        ax.axhline(y=target_acc, color='#1B4965', linestyle=':', linewidth=1.5,
                   label=f'CoT 3-shot ({target_acc:.1f}%)')

    ax.set_xlabel('Scaling Factor (α)', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title('FSV Performance vs Scaling Factor', fontsize=10)
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    all_accs = []
    for layer in fsv_results:
        for scale in fsv_results[layer]:
            all_accs.append(fsv_results[layer][scale]["accuracy"])
    if baselines:
        all_accs.extend([b.get("accuracy", 0) for b in baselines.values()])

    y_min = max(0, min(all_accs) - 3)
    y_max = min(100, max(all_accs) + 3)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs scale plot to: {output_path}")


def plot_comparison_bars(fsv_results: dict, baselines: dict, output_path: Path):
    """Plot bar chart comparing 0-shot, 0-shot+FSV, and 3-shot."""
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.7))

    # Get best FSV config (layer 12, scale 0.2)
    best_layer = 12
    best_scale = 0.2

    if best_layer in fsv_results and best_scale in fsv_results[best_layer]:
        fsv_data = fsv_results[best_layer][best_scale]
    else:
        print(f"Warning: Could not find FSV data for layer {best_layer}, scale {best_scale}")
        return

    # Get baseline data (use defaults if not found)
    cot_0shot = baselines.get("cot_0shot", {"accuracy": 29.14, "input_tokens": 1215, "latency": 68.7})
    cot_3shot = baselines.get("cot_3shot", {"accuracy": 32.08, "input_tokens": 5257, "latency": 93.3})

    labels = ['CoT 0-shot', f'0-shot + FSV\n(L{best_layer}, α={best_scale})', 'CoT 3-shot']

    # Plot 1: Accuracy
    ax1 = axes[0]
    accuracies = [cot_0shot["accuracy"], fsv_data["accuracy"], cot_3shot["accuracy"]]
    colors = ['#2E86AB', '#E94F37', '#1B4965']

    bars1 = ax1.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax1.set_ylabel('Accuracy (%)', fontsize=9)
    ax1.set_title('(a) Accuracy', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', labelsize=7)

    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Set y-axis to start from a reasonable value
    ax1.set_ylim(min(accuracies) - 5, max(accuracies) + 5)

    # Plot 2: Input Tokens
    ax2 = axes[1]
    # FSV uses 0-shot tokens, so same as cot_0shot
    tokens = [cot_0shot.get("input_tokens", 1215),
              cot_0shot.get("input_tokens", 1215),  # FSV uses 0-shot prompt length
              cot_3shot.get("input_tokens", 5257)]

    bars2 = ax2.bar(labels, tokens, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.set_ylabel('Avg Input Tokens', fontsize=9)
    ax2.set_title('(b) Input Token Usage', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', labelsize=7)

    for bar, tok in zip(bars2, tokens):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{tok:.0f}', ha='center', va='bottom', fontsize=8)

    # Plot 3: Latency
    ax3 = axes[2]
    latencies = [cot_0shot.get("latency", 68.7),
                 fsv_data.get("latency", 170),  # FSV latency
                 cot_3shot.get("latency", 93.3)]

    bars3 = ax3.bar(labels, latencies, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax3.set_ylabel('Total Latency (s)', fontsize=9)
    ax3.set_title('(c) Inference Latency', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', labelsize=7)

    for bar, lat in zip(bars3, latencies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{lat:.1f}s', ha='center', va='bottom', fontsize=8)

    plt.suptitle('FSV vs ICL Comparison (Qwen2.5-1.5B-Instruct)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison bars to: {output_path}")


def plot_combined_fsv(fsv_results: dict, baselines: dict, output_path: Path):
    """Create combined figure with (a) accuracy vs alpha, (b) comparison bars."""
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.7))

    # Get baseline values (use defaults if not found)
    cot_0shot = baselines.get("cot_0shot", {"accuracy": 29.14, "input_tokens": 1215, "latency": 68.7})
    cot_3shot = baselines.get("cot_3shot", {"accuracy": 32.08, "input_tokens": 5257, "latency": 93.3})

    # Plot (a): Accuracy vs Scale
    ax1 = axes[0]
    colors = {12: '#E94F37', 16: '#2E86AB'}
    markers = {12: 'o', 16: 's'}

    for layer in sorted(fsv_results.keys()):
        if layer not in [12, 16]:
            continue

        scales = sorted(fsv_results[layer].keys())
        accs = [fsv_results[layer][s]["accuracy"] for s in scales]

        ax1.plot(scales, accs, marker=markers[layer], linewidth=1.5, markersize=5,
                 color=colors[layer], label=f'Layer {layer}')

    # Baseline lines
    ax1.axhline(y=cot_0shot["accuracy"], color='gray', linestyle='--', linewidth=1.5,
                label=f'CoT 0-shot ({cot_0shot["accuracy"]:.1f}%)')
    ax1.axhline(y=cot_3shot["accuracy"], color='#1B4965', linestyle=':', linewidth=1.5,
                label=f'CoT 3-shot ({cot_3shot["accuracy"]:.1f}%)')

    ax1.set_xlabel('Scaling Factor (α)', fontsize=9)
    ax1.set_ylabel('Accuracy (%)', fontsize=9)
    ax1.set_title('(a) FSV Accuracy vs Scaling Factor', fontsize=10)
    ax1.legend(loc='lower left', fontsize=7)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, alpha=0.3)

    # Set y-limits
    all_accs = []
    for layer in fsv_results:
        for scale in fsv_results[layer]:
            all_accs.append(fsv_results[layer][scale]["accuracy"])
    all_accs.extend([cot_0shot["accuracy"], cot_3shot["accuracy"]])
    ax1.set_ylim(min(all_accs) - 2, max(all_accs) + 2)

    # Plot (b): Comparison bars (accuracy only)
    ax2 = axes[1]

    # Get best FSV config
    best_layer = 12
    best_scale = 0.2
    fsv_acc = fsv_results.get(best_layer, {}).get(best_scale, {}).get("accuracy", 31.0)

    labels = ['CoT\n0-shot', f'0-shot + FSV\n(L{best_layer}, α={best_scale})', 'CoT\n3-shot']
    accuracies = [cot_0shot["accuracy"], fsv_acc, cot_3shot["accuracy"]]
    bar_colors = ['#2E86AB', '#E94F37', '#1B4965']

    bars = ax2.bar(labels, accuracies, color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Accuracy (%)', fontsize=9)
    ax2.set_title('(b) Method Comparison', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', labelsize=7)
    ax2.set_ylim(min(accuracies) - 3, max(accuracies) + 3)

    # Add annotation for improvement
    delta = fsv_acc - cot_0shot["accuracy"]
    ax2.annotate(f'+{delta:.1f}%', xy=(1, fsv_acc), xytext=(1.3, fsv_acc + 1),
                 fontsize=8, color='#E94F37', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.2))

    plt.suptitle('Financial Steering Vector (FSV) Results\n(Qwen2.5-1.5B-Instruct, 429 samples)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined FSV plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate FSV plots from experiment results")
    parser.add_argument("--outputs_sv_dir", type=str, default="outputs_SV",
                        help="Path to outputs_SV directory")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Path to outputs directory for baselines")
    parser.add_argument("--report_dir", type=str, default="figures/2_steering_vector",
                        help="Path to save figures")
    args = parser.parse_args()

    base_dir = Path("/home/yeopjin/orcd/pool/workspace/Financial_task_vector")
    outputs_sv_dir = base_dir / args.outputs_sv_dir
    outputs_dir = base_dir / args.outputs_dir
    report_dir = base_dir / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load FSV results
    print("Loading FSV results...")
    fsv_results = load_fsv_results(outputs_sv_dir)

    print(f"Found FSV results for layers: {list(fsv_results.keys())}")
    for layer, scales in fsv_results.items():
        print(f"  Layer {layer}: scales {list(scales.keys())}")

    # Load baseline results
    print("Loading baseline results...")
    baselines = load_baseline_results(outputs_dir)
    print(f"Found baselines: {list(baselines.keys())}")

    if not fsv_results:
        print("No FSV results found!")
        return

    # Generate plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_accuracy_vs_scale(fsv_results, baselines,
                           report_dir / f"fsv_accuracy_vs_scale_{timestamp}.png")
    plot_comparison_bars(fsv_results, baselines,
                         report_dir / f"fsv_comparison_bars_{timestamp}.png")
    plot_combined_fsv(fsv_results, baselines,
                      report_dir / f"fsv_combined_{timestamp}.png")

    print(f"\nDone! Figures saved to: {report_dir}")


if __name__ == "__main__":
    main()
