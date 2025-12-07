#!/usr/bin/env python3
"""
Generate publication-quality figures from Gemini reproduction experiment results.
Creates visualizations suitable for academic report and presentation.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette (colorblind-friendly)
COLORS = {
    'disk': '#E74C3C',      # Red
    'ram': '#2ECC71',       # Green  
    'gemini': '#3498DB',    # Blue
    'baseline': '#95A5A6',  # Gray
}


def load_results(results_dir: Path):
    """Load the aggregated results file."""
    # Find the aggregated results file
    agg_files = list(results_dir.glob("aggregated_*.json"))
    if not agg_files:
        raise FileNotFoundError("No aggregated results file found in results/")
    
    latest = sorted(agg_files)[-1]
    print(f"Loading results from: {latest}")
    
    with open(latest) as f:
        return json.load(f)


def figure1_checkpoint_time_comparison(data: dict, output_dir: Path):
    """
    Figure 1: Bar chart comparing checkpoint save times.
    Disk-based vs RAM-based (single GPU).
    """
    agg = data['aggregated']
    
    disk_mean = agg['baseline']['avg_checkpoint_ms_mean']
    disk_std = agg['baseline']['avg_checkpoint_ms_std']
    ram_mean = agg['gemini_single']['avg_checkpoint_ms_mean']
    ram_std = agg['gemini_single']['avg_checkpoint_ms_std']
    
    # Calculate speedup
    speedup = disk_mean / ram_mean
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    x = np.array([0, 1])
    heights = [disk_mean, ram_mean]
    errors = [disk_std, ram_std]
    colors = [COLORS['disk'], COLORS['ram']]
    labels = ['Disk-based\n(Baseline)', 'RAM-based\n(Gemini)']
    
    bars = ax.bar(x, heights, yerr=errors, capsize=8, color=colors, 
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('Checkpoint Time (ms)')
    ax.set_title('Single-GPU Checkpoint Save Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, disk_mean * 1.3)
    
    # Add value labels on bars
    for bar, h, e in zip(bars, heights, errors):
        ax.text(bar.get_x() + bar.get_width()/2, h + e + 100, 
                f'{h:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Add speedup annotation
    ax.annotate(f'{speedup:.1f}× faster', 
                xy=(1, ram_mean), xytext=(0.5, disk_mean * 0.6),
                fontsize=14, fontweight='bold', color=COLORS['gemini'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gemini'], lw=2),
                ha='center')
    
    plt.tight_layout()
    output_path = output_dir / 'fig1_checkpoint_time.png'
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)
    
    return speedup


def figure2_recovery_time_comparison(data: dict, output_dir: Path):
    """
    Figure 2: Bar chart comparing recovery times.
    Disk vs RAM for single-GPU scenario.
    """
    agg = data['aggregated']
    
    disk_mean = agg['baseline']['disk_recovery_ms_mean']
    disk_std = agg['baseline']['disk_recovery_ms_std']
    ram_mean = agg['gemini_single']['ram_recovery_ms_mean']
    ram_std = agg['gemini_single']['ram_recovery_ms_std']
    
    speedup = disk_mean / ram_mean
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    x = np.array([0, 1])
    heights = [disk_mean, ram_mean]
    errors = [disk_std, ram_std]
    colors = [COLORS['disk'], COLORS['ram']]
    labels = ['Disk Recovery\n(Baseline)', 'RAM Recovery\n(Gemini)']
    
    bars = ax.bar(x, heights, yerr=errors, capsize=8, color=colors,
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('Recovery Time (ms)')
    ax.set_title('Single-GPU Recovery Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, disk_mean * 1.4)
    
    # Add value labels
    for bar, h, e in zip(bars, heights, errors):
        ax.text(bar.get_x() + bar.get_width()/2, h + e + 20,
                f'{h:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Add speedup annotation
    ax.annotate(f'{speedup:.1f}× faster',
                xy=(1, ram_mean), xytext=(0.5, disk_mean * 0.5),
                fontsize=14, fontweight='bold', color=COLORS['gemini'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gemini'], lw=2),
                ha='center')
    
    plt.tight_layout()
    output_path = output_dir / 'fig2_recovery_time.png'
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)
    
    return speedup


def figure3_multi_gpu_comparison(data: dict, output_dir: Path):
    """
    Figure 3: Grouped bar chart for multi-GPU results.
    Comparing checkpoint and recovery times across configurations.
    """
    agg = data['aggregated']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Multi-GPU Checkpoint Times
    ax = axes[0]
    x = np.array([0, 1])
    disk_ckpt = agg['multi_disk']['avg_checkpoint_ms_mean']
    disk_ckpt_std = agg['multi_disk']['avg_checkpoint_ms_std']
    # For Gemini multi, use local save time (not including replication)
    ram_save = agg['comparison']['multi_gpu_ram_save_ms_mean']
    ram_save_std = agg['comparison']['multi_gpu_ram_save_ms_std']
    
    bars = ax.bar(x, [disk_ckpt, ram_save], 
                  yerr=[disk_ckpt_std, ram_save_std],
                  capsize=8, color=[COLORS['disk'], COLORS['ram']],
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('Checkpoint Time (ms)')
    ax.set_title('Multi-GPU (4 GPUs) Checkpoint Time')
    ax.set_xticks(x)
    ax.set_xticklabels(['Disk-based', 'RAM-based\n(local save)'])
    
    speedup = disk_ckpt / ram_save
    for bar, h in zip(bars, [disk_ckpt, ram_save]):
        ax.text(bar.get_x() + bar.get_width()/2, h + 150,
                f'{h:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    ax.annotate(f'{speedup:.1f}× faster',
                xy=(1, ram_save), xytext=(0.5, disk_ckpt * 0.5),
                fontsize=12, fontweight='bold', color=COLORS['gemini'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gemini'], lw=2),
                ha='center')
    
    # Right: Multi-GPU Recovery Times
    ax = axes[1]
    disk_rec = agg['multi_disk']['measured_recovery_ms_mean']
    disk_rec_std = agg['multi_disk']['measured_recovery_ms_std']
    ram_rec = agg['gemini_multi']['measured_recovery_ms_mean']
    ram_rec_std = agg['gemini_multi']['measured_recovery_ms_std']
    
    bars = ax.bar(x, [disk_rec, ram_rec],
                  yerr=[disk_rec_std, ram_rec_std],
                  capsize=8, color=[COLORS['disk'], COLORS['ram']],
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('Recovery Time (ms)')
    ax.set_title('Multi-GPU (4 GPUs) Recovery Time')
    ax.set_xticks(x)
    ax.set_xticklabels(['Disk-based', 'RAM-based'])
    
    speedup = disk_rec / ram_rec
    for bar, h in zip(bars, [disk_rec, ram_rec]):
        ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                f'{h:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    ax.annotate(f'{speedup:.1f}× faster',
                xy=(1, ram_rec), xytext=(0.5, disk_rec * 0.5),
                fontsize=12, fontweight='bold', color=COLORS['gemini'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gemini'], lw=2),
                ha='center')
    
    plt.tight_layout()
    output_path = output_dir / 'fig3_multi_gpu_comparison.png'
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def figure4_throughput_comparison(data: dict, output_dir: Path):
    """
    Figure 4: Throughput comparison showing minimal overhead.
    """
    agg = data['aggregated']
    
    baseline_tp = agg['baseline']['throughput_mean']
    baseline_tp_std = agg['baseline']['throughput_std']
    gemini_tp = agg['gemini_single']['throughput_mean']
    gemini_tp_std = agg['gemini_single']['throughput_std']
    
    overhead = ((baseline_tp - gemini_tp) / baseline_tp) * 100 if gemini_tp < baseline_tp else 0
    improvement = ((gemini_tp - baseline_tp) / baseline_tp) * 100
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    x = np.array([0, 1])
    heights = [baseline_tp, gemini_tp]
    errors = [baseline_tp_std, gemini_tp_std]
    colors = [COLORS['baseline'], COLORS['gemini']]
    labels = ['Disk Checkpointing\n(Baseline)', 'RAM Checkpointing\n(Gemini)']
    
    bars = ax.bar(x, heights, yerr=errors, capsize=8, color=colors,
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('Throughput (iterations/sec)')
    ax.set_title('Training Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Set y-axis to start from 0 for proper visualization
    y_max = max(heights) * 1.25
    ax.set_ylim(0, y_max)
    
    # Add value labels on bars
    for bar, h, e in zip(bars, heights, errors):
        ax.text(bar.get_x() + bar.get_width()/2, h + e + 1,
                f'{h:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation - use arrow to connect the two bars
    if improvement > 0:
        ax.annotate(f'+{improvement:.0f}% faster',
                    xy=(1, gemini_tp), xytext=(0.5, gemini_tp * 0.7),
                    fontsize=14, fontweight='bold', color=COLORS['gemini'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['gemini'], lw=2),
                    ha='center')
    
    plt.tight_layout()
    output_path = output_dir / 'fig4_throughput.png'
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)
    
    return improvement


def figure5_summary_speedups(data: dict, output_dir: Path):
    """
    Figure 5: Summary bar chart of all speedup factors.
    Great for presentations.
    """
    agg = data['aggregated']
    comp = agg['comparison']
    
    # Collect speedups
    categories = [
        'Single-GPU\nCheckpoint',
        'Single-GPU\nRecovery', 
        'Multi-GPU\nRecovery'
    ]
    
    speedups = [
        comp['single_gpu_speedup_mean'],
        comp['recovery_speedup_mean'],
        comp['multi_gpu_recovery_speedup_mean']
    ]
    
    stds = [
        comp['single_gpu_speedup_std'],
        comp['recovery_speedup_std'],
        comp['multi_gpu_recovery_speedup_std']
    ]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(categories))
    bars = ax.bar(x, speedups, yerr=stds, capsize=8, 
                  color=[COLORS['gemini']] * len(categories),
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('Speedup Factor (×)')
    ax.set_title('Gemini In-Memory Checkpointing: Performance Speedups')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    
    # Add horizontal reference line at 10x (the target)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target: 10× speedup')
    ax.axhline(y=13, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target: 13× speedup')
    
    # Add value labels
    for bar, h, std in zip(bars, speedups, stds):
        ax.text(bar.get_x() + bar.get_width()/2, h + std + 0.3,
                f'{h:.1f}×', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(speedups) * 1.3)
    
    plt.tight_layout()
    output_path = output_dir / 'fig5_speedup_summary.png'
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def figure6_breakdown_stacked(data: dict, output_dir: Path):
    """
    Figure 6: Stacked bar showing checkpoint time breakdown.
    Local save vs replication time for multi-GPU Gemini.
    """
    agg = data['aggregated']
    
    # Get multi-GPU data
    local_save = agg['comparison']['multi_gpu_ram_save_ms_mean']
    replication = agg['comparison']['multi_gpu_replication_ms_mean']
    total_gemini = agg['gemini_multi']['avg_checkpoint_ms_mean']
    disk_total = agg['multi_disk']['avg_checkpoint_ms_mean']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.array([0, 1])
    width = 0.5
    
    # Disk: single bar
    ax.bar(0, disk_total, width, color=COLORS['disk'], edgecolor='black', 
           linewidth=1.5, label='Disk I/O')
    
    # Gemini: stacked bar
    ax.bar(1, local_save, width, color=COLORS['ram'], edgecolor='black',
           linewidth=1.5, label='Local RAM Save')
    ax.bar(1, replication, width, bottom=local_save, color='#9B59B6', 
           edgecolor='black', linewidth=1.5, label='Network Replication')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('Multi-GPU Checkpoint Time Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(['Disk-based\nCheckpoint', 'Gemini\n(RAM + Replication)'])
    ax.legend()
    
    # Add annotations
    ax.text(0, disk_total + 200, f'{disk_total:.0f}ms', ha='center', fontweight='bold')
    ax.text(1, total_gemini + 200, f'{total_gemini:.0f}ms\n(total)', ha='center', fontweight='bold')
    ax.text(1, local_save/2, f'{local_save:.0f}ms', ha='center', va='center', 
            color='white', fontweight='bold')
    ax.text(1, local_save + replication/2, f'{replication:.0f}ms', ha='center', 
            va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'fig6_checkpoint_breakdown.png'
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def create_results_table(data: dict, output_dir: Path):
    """Create a summary table as text file for the report."""
    agg = data['aggregated']
    comp = agg['comparison']
    
    table = """
================================================================================
                    EXPERIMENT RESULTS SUMMARY
================================================================================

Configuration:
  - Iterations: 100
  - Checkpoint Frequency: Every 25 iterations
  - Model: 12-layer Transformer (1024 hidden, 16 heads)
  - Number of Runs: 3 (averaged)

--------------------------------------------------------------------------------
SINGLE-GPU RESULTS
--------------------------------------------------------------------------------
                          Disk-based          RAM-based (Gemini)     Speedup
Checkpoint Save Time      {disk_ckpt:.0f} +/- {disk_ckpt_std:.0f} ms    {ram_ckpt:.0f} +/- {ram_ckpt_std:.0f} ms          {ckpt_speedup:.1f}x
Recovery Time             {disk_rec:.0f} +/- {disk_rec_std:.0f} ms     {ram_rec:.0f} +/- {ram_rec_std:.0f} ms           {rec_speedup:.1f}x
Throughput (iter/sec)     {disk_tp:.1f} +/- {disk_tp_std:.1f}        {ram_tp:.1f} +/- {ram_tp_std:.1f}          +{tp_gain:.0f}%

--------------------------------------------------------------------------------
MULTI-GPU RESULTS (4 GPUs)
--------------------------------------------------------------------------------
                          Disk-based          RAM-based (Gemini)     Speedup
Checkpoint Save Time      {mdisk_ckpt:.0f} +/- {mdisk_ckpt_std:.0f} ms   {mram_ckpt:.0f} +/- {mram_ckpt_std:.0f} ms (local)  {m_ckpt_speedup:.1f}x
Recovery Time             {mdisk_rec:.0f} +/- {mdisk_rec_std:.0f} ms    {mram_rec:.0f} +/- {mram_rec_std:.0f} ms           {m_rec_speedup:.1f}x

--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------
- Single-GPU checkpoint speedup: {ckpt_speedup:.1f}x (target: 10-15x)
- Single-GPU recovery speedup: {rec_speedup:.1f}x (target: 10-13x)
- Multi-GPU recovery speedup: {m_rec_speedup:.1f}x
- Throughput improvement: +{tp_gain:.0f}%
- Memory overhead: ~{mem_mb:.0f} MB per checkpoint

================================================================================
""".format(
        disk_ckpt=agg['baseline']['avg_checkpoint_ms_mean'],
        disk_ckpt_std=agg['baseline']['avg_checkpoint_ms_std'],
        ram_ckpt=agg['gemini_single']['avg_checkpoint_ms_mean'],
        ram_ckpt_std=agg['gemini_single']['avg_checkpoint_ms_std'],
        ckpt_speedup=comp['single_gpu_speedup_mean'],
        disk_rec=agg['baseline']['disk_recovery_ms_mean'],
        disk_rec_std=agg['baseline']['disk_recovery_ms_std'],
        ram_rec=agg['gemini_single']['ram_recovery_ms_mean'],
        ram_rec_std=agg['gemini_single']['ram_recovery_ms_std'],
        rec_speedup=comp['recovery_speedup_mean'],
        disk_tp=agg['baseline']['throughput_mean'],
        disk_tp_std=agg['baseline']['throughput_std'],
        ram_tp=agg['gemini_single']['throughput_mean'],
        ram_tp_std=agg['gemini_single']['throughput_std'],
        tp_gain=((agg['gemini_single']['throughput_mean'] - agg['baseline']['throughput_mean']) / 
                 agg['baseline']['throughput_mean'] * 100),
        mdisk_ckpt=agg['multi_disk']['avg_checkpoint_ms_mean'],
        mdisk_ckpt_std=agg['multi_disk']['avg_checkpoint_ms_std'],
        mram_ckpt=comp['multi_gpu_ram_save_ms_mean'],
        mram_ckpt_std=comp['multi_gpu_ram_save_ms_std'],
        m_ckpt_speedup=agg['multi_disk']['avg_checkpoint_ms_mean'] / comp['multi_gpu_ram_save_ms_mean'],
        mdisk_rec=agg['multi_disk']['measured_recovery_ms_mean'],
        mdisk_rec_std=agg['multi_disk']['measured_recovery_ms_std'],
        mram_rec=agg['gemini_multi']['measured_recovery_ms_mean'],
        mram_rec_std=agg['gemini_multi']['measured_recovery_ms_std'],
        m_rec_speedup=comp['multi_gpu_recovery_speedup_mean'],
        mem_mb=agg['gemini_single']['memory_mb_mean']
    )
    
    output_path = output_dir / 'results_summary.txt'
    with open(output_path, 'w') as f:
        f.write(table)
    print(f"Saved: {output_path}")
    print(table)


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / 'results'
    figures_dir = project_dir / 'figures'
    
    # Create figures directory
    figures_dir.mkdir(exist_ok=True)
    
    # Load data
    print("=" * 60)
    print("GENERATING FIGURES FOR CS240 GEMINI REPRODUCTION PROJECT")
    print("=" * 60)
    
    try:
        data = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nConfiguration: {data['config']}")
    print(f"Number of runs: {data['num_runs']}")
    print()
    
    # Generate all figures
    print("\nGenerating figures...")
    print("-" * 40)
    
    ckpt_speedup = figure1_checkpoint_time_comparison(data, figures_dir)
    rec_speedup = figure2_recovery_time_comparison(data, figures_dir)
    figure3_multi_gpu_comparison(data, figures_dir)
    tp_improvement = figure4_throughput_comparison(data, figures_dir)
    figure5_summary_speedups(data, figures_dir)
    figure6_breakdown_stacked(data, figures_dir)
    
    # Create results table
    print("-" * 40)
    create_results_table(data, figures_dir)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Checkpoint speedup: {ckpt_speedup:.1f}x")
    print(f"Recovery speedup: {rec_speedup:.1f}x")
    print(f"Throughput improvement: +{tp_improvement:.1f}%")
    print(f"\nAll figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
