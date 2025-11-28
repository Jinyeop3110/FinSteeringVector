#!/usr/bin/env python3
"""Generate comparison plot for FinQA experiments with different prompting strategies."""

import matplotlib.pyplot as plt
import numpy as np

# Results from experiments (16 reps each for CoT 1-4 shot)
# Vanilla and CoT 0-shot don't need multiple reps (no ICL randomness)
results = {
    'Vanilla': {'mean': 8.39, 'std': 0.0},
    'CoT 0-shot': {'mean': 29.37, 'std': 0.0},
    'CoT 1-shot': {'mean': 31.37, 'std': 1.12},
    'CoT 2-shot': {'mean': 32.01, 'std': 1.08},
    'CoT 3-shot': {'mean': 32.08, 'std': 0.91},
    'CoT 4-shot': {'mean': 30.00, 'std': 0.85},
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Data for plotting
labels = list(results.keys())
means = [results[k]['mean'] for k in labels]
stds = [results[k]['std'] for k in labels]

# Colors - vanilla in red, CoT 0-shot in orange, others in blue gradient
colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2980b9', '#3498db', '#85c1e9']

# Create bar plot with error bars
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)

# Customize plot
ax.set_xlabel('Prompting Strategy', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Qwen2.5-1.5B-Instruct on FinQA: Prompting Strategy Comparison\n(Full Context, 16 reps for ICL variants)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylim(0, 40)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    if std > 0:
        label = f'{mean:.2f}%\n±{std:.2f}'
    else:
        label = f'{mean:.2f}%'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Add horizontal line for reference
ax.axhline(y=results['CoT 0-shot']['mean'], color='#ff7f0e', linestyle='--', alpha=0.5, label='CoT 0-shot baseline')

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('/home/yeopjin/orcd/pool/workspace/Financial_task_vector/finqa_prompting_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('/home/yeopjin/orcd/pool/workspace/Financial_task_vector/finqa_prompting_comparison.pdf', bbox_inches='tight')
print("Plot saved to finqa_prompting_comparison.png and finqa_prompting_comparison.pdf")

# Print summary table
print("\n" + "="*60)
print("SUMMARY: Qwen2.5-1.5B-Instruct on FinQA (Full Context)")
print("="*60)
print(f"{'Strategy':<15} {'Mean Acc':<12} {'Std':<10} {'n_reps'}")
print("-"*60)
for name, data in results.items():
    n_reps = 16 if data['std'] > 0 else 1
    print(f"{name:<15} {data['mean']:>7.2f}%     {data['std']:>5.2f}%     {n_reps}")
print("="*60)
print("\nKey Findings:")
print("- CoT prompting provides significant boost over vanilla (+23%)")
print("- 2-3 shot performs best (32.01-32.08%)")
print("- Performance degrades at 4-shot, likely due to context limitations")
print("- Optimal strategy: CoT 3-shot (32.08% ± 0.91%)")
