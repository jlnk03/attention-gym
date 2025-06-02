#!/usr/bin/env python3
"""
Visualization script for attention benchmark results.

This script reads the unified JSON file created by benchmark.py and generates
various interesting visualizations comparing different runs and examples.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def load_benchmark_data(json_file: str = "benchmark_results.json") -> Dict:
    """Load benchmark results from the JSON file."""
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Benchmark results file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def get_available_examples(data: Dict) -> List[str]:
    """Get list of available example names."""
    return list(data.keys())


def get_available_runs(data: Dict, example: str) -> List[str]:
    """Get list of available run IDs for an example."""
    if example not in data:
        return []
    return list(data[example].get("runs", {}).keys())


def prepare_data_for_plotting(data: Dict, example_name: str, metric: str = "fw_tflops") -> pd.DataFrame:
    """Prepare data for plotting by converting it to a pandas DataFrame."""
    if example_name not in data:
        raise KeyError(f"Example {example_name} not found in benchmark data")
    
    example_data = data[example_name]
    latest_run = example_data["latest"]
    results = latest_run["results"]
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Add metadata columns
    df["timestamp"] = latest_run["timestamp"]
    df["device"] = latest_run["metadata"]["device"]
    df["data_type"] = latest_run["metadata"]["data_type"]
    
    return df


def plot_benchmark_results(
    data: Dict,
    example_name: str,
    metric: str = "fw_tflops",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """Plot benchmark results for a specific example and metric."""
    df = prepare_data_for_plotting(data, example_name, metric)
    
    # Set up the plot style - use a valid style
    plt.style.use('default')
    sns.set_theme()  # Apply seaborn styling
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars with gradient colors
    colors = sns.color_palette("viridis", n_colors=len(df))
    bars = ax.bar(df["operation"], df[metric], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(df[metric]) * 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
    
    # Customize the plot
    ax.set_title(f"{example_name.replace('_', ' ').title()} - {get_metric_label(metric)}", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(get_metric_label(metric), fontsize=12, fontweight='bold')
    ax.set_xlabel("Operation", fontsize=12, fontweight='bold')
    
    # Improve x-axis labels
    ax.set_xticks(range(len(df["operation"])))
    ax.set_xticklabels(df["operation"], rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add metadata as a styled text box
    metadata_text = (
        f"Device: {df['device'].iloc[0]}\n"
        f"Data Type: {df['data_type'].iloc[0]}\n"
        f"Timestamp: {df['timestamp'].iloc[0][:19]}"
    )
    
    # Create a text box with background
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_examples(
    data: Dict,
    metrics: List[str] = ["fw_tflops", "bw_tflops"],
    output_dir: str = "benchmark_plots"
):
    """Plot all examples and metrics, saving them to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for example_name in data.keys():
        for metric in metrics:
            save_path = output_path / f"{example_name}_{metric}.png"
            plot_benchmark_results(
                data,
                example_name,
                metric,
                save_path=str(save_path)
            )


def create_comparison_plot(data: Dict, examples: List[str], metric: str = "fw_time", 
                          output_dir: str = "benchmark_plots", run_id: Optional[str] = None):
    """
    Create a comparison plot for a specific metric across examples.
    
    Args:
        data: Benchmark data dictionary
        examples: List of example names to compare
        metric: Metric to plot ('fw_time', 'bw_time', 'fw_tflops', 'bw_tflops')
        output_dir: Directory to save plots
        run_id: Specific run ID to use, or None for latest
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    all_operations = set()
    plot_data = {}
    
    # Collect data for each example
    for example in examples:
        if example not in data:
            print(f"Warning: Example '{example}' not found in data")
            continue
        
        if run_id is None:
            # Use latest run
            if "latest" not in data[example]:
                print(f"Warning: No latest run found for example '{example}'")
                continue
            results = data[example]["latest"]["results"]
        else:
            # Use specific run
            if run_id not in data[example].get("runs", {}):
                print(f"Warning: Run ID '{run_id}' not found for example '{example}'")
                continue
            results = data[example]["runs"][run_id]["results"]
        
        # Extract metric values
        operations = [r["operation"] for r in results]
        values = [r[metric] for r in results]
        
        all_operations.update(operations)
        plot_data[example] = dict(zip(operations, values))
    
    # Prepare data for plotting
    operations = sorted(all_operations)
    x = np.arange(len(operations))
    width = 0.8 / len(examples)
    
    # Create bars for each example
    for i, example in enumerate(examples):
        if example not in plot_data:
            continue
        
        values = [plot_data[example].get(op, 0) for op in operations]
        plt.bar(x + i * width, values, width, label=example.replace("_", " ").title())
    
    plt.xlabel('Operations')
    plt.ylabel(get_metric_label(metric))
    plt.title(f'{get_metric_label(metric)} Comparison Across Examples')
    plt.xticks(x + width * (len(examples) - 1) / 2, operations, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = f"comparison_{metric}_{'_'.join(examples)}.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path / filename}")
    plt.close()


def create_time_series_plot(data: Dict, example: str, operation: str, 
                           output_dir: str = "benchmark_plots"):
    """
    Create a time series plot showing performance over different runs.
    
    Args:
        data: Benchmark data dictionary
        example: Example name
        operation: Operation name (e.g., 'flexattention', 'F.sdpa + mask')
        output_dir: Directory to save plots
    """
    if example not in data:
        print(f"Warning: Example '{example}' not found in data")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    runs = data[example].get("runs", {})
    if not runs:
        print(f"Warning: No runs found for example '{example}'")
        return
    
    # Extract data for time series
    timestamps = []
    fw_times = []
    bw_times = []
    fw_tflops = []
    bw_tflops = []
    
    for run_id, run_data in sorted(runs.items()):
        timestamp = datetime.fromisoformat(run_data["timestamp"])
        results = run_data["results"]
        
        # Find the specific operation
        op_data = None
        for result in results:
            if result["operation"] == operation:
                op_data = result
                break
        
        if op_data:
            timestamps.append(timestamp)
            fw_times.append(op_data["fw_time"])
            bw_times.append(op_data["bw_time"])
            fw_tflops.append(op_data["fw_tflops"])
            bw_tflops.append(op_data["bw_tflops"])
    
    if not timestamps:
        print(f"Warning: No data found for operation '{operation}' in example '{example}'")
        return
    
    # Create subplot with 2x2 layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Forward time
    ax1.plot(timestamps, fw_times, 'o-', color='blue', linewidth=2, markersize=6)
    ax1.set_title(f'Forward Time - {operation}')
    ax1.set_ylabel('Time (ms)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Backward time
    ax2.plot(timestamps, bw_times, 'o-', color='red', linewidth=2, markersize=6)
    ax2.set_title(f'Backward Time - {operation}')
    ax2.set_ylabel('Time (ms)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Forward TFLOPS
    ax3.plot(timestamps, fw_tflops, 'o-', color='green', linewidth=2, markersize=6)
    ax3.set_title(f'Forward TFLOPS - {operation}')
    ax3.set_ylabel('TFLOPS')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Backward TFLOPS
    ax4.plot(timestamps, bw_tflops, 'o-', color='orange', linewidth=2, markersize=6)
    ax4.set_title(f'Backward TFLOPS - {operation}')
    ax4.set_ylabel('TFLOPS')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Performance Over Time: {example.replace("_", " ").title()}')
    plt.tight_layout()
    
    filename = f"timeseries_{example}_{operation.replace(' ', '_').replace('+', 'plus')}.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved time series plot: {output_path / filename}")
    plt.close()


def create_performance_matrix(data: Dict, examples: List[str], 
                            output_dir: str = "benchmark_plots", 
                            run_id: Optional[str] = None):
    """
    Create a heatmap matrix showing performance across examples and operations.
    
    Args:
        data: Benchmark data dictionary
        examples: List of example names
        output_dir: Directory to save plots
        run_id: Specific run ID to use, or None for latest
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all operations and create matrices
    all_operations = set()
    fw_time_data = {}
    fw_tflops_data = {}
    
    for example in examples:
        if example not in data:
            continue
        
        if run_id is None:
            if "latest" not in data[example]:
                continue
            results = data[example]["latest"]["results"]
        else:
            if run_id not in data[example].get("runs", {}):
                continue
            results = data[example]["runs"][run_id]["results"]
        
        fw_time_data[example] = {}
        fw_tflops_data[example] = {}
        
        for result in results:
            operation = result["operation"]
            all_operations.add(operation)
            fw_time_data[example][operation] = result["fw_time"]
            fw_tflops_data[example][operation] = result["fw_tflops"]
    
    operations = sorted(all_operations)
    
    # Create matrices
    fw_time_matrix = []
    fw_tflops_matrix = []
    
    for example in examples:
        if example in fw_time_data:
            fw_time_row = [fw_time_data[example].get(op, np.nan) for op in operations]
            fw_tflops_row = [fw_tflops_data[example].get(op, np.nan) for op in operations]
        else:
            fw_time_row = [np.nan] * len(operations)
            fw_tflops_row = [np.nan] * len(operations)
        
        fw_time_matrix.append(fw_time_row)
        fw_tflops_matrix.append(fw_tflops_row)
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Forward time heatmap
    sns.heatmap(fw_time_matrix, 
                xticklabels=operations, 
                yticklabels=[ex.replace("_", " ").title() for ex in examples],
                annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Time (ms)'})
    ax1.set_title('Forward Time Heatmap')
    ax1.set_xlabel('Operations')
    ax1.set_ylabel('Examples')
    
    # Forward TFLOPS heatmap
    sns.heatmap(fw_tflops_matrix, 
                xticklabels=operations, 
                yticklabels=[ex.replace("_", " ").title() for ex in examples],
                annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'TFLOPS'})
    ax2.set_title('Forward TFLOPS Heatmap')
    ax2.set_xlabel('Operations')
    ax2.set_ylabel('Examples')
    
    plt.tight_layout()
    
    filename = f"performance_matrix_{'_'.join(examples)}.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved performance matrix: {output_path / filename}")
    plt.close()


def create_speedup_analysis(data: Dict, examples: List[str], baseline_op: str = "F.sdpa + mask",
                           target_op: str = "flexattention", output_dir: str = "benchmark_plots",
                           run_id: Optional[str] = None):
    """
    Create speedup analysis comparing target operation to baseline.
    
    Args:
        data: Benchmark data dictionary
        examples: List of example names
        baseline_op: Baseline operation name
        target_op: Target operation name
        output_dir: Directory to save plots
        run_id: Specific run ID to use, or None for latest
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    speedups_fw = []
    speedups_bw = []
    example_names = []
    
    for example in examples:
        if example not in data:
            continue
        
        if run_id is None:
            if "latest" not in data[example]:
                continue
            results = data[example]["latest"]["results"]
        else:
            if run_id not in data[example].get("runs", {}):
                continue
            results = data[example]["runs"][run_id]["results"]
        
        baseline_data = None
        target_data = None
        
        for result in results:
            if result["operation"] == baseline_op:
                baseline_data = result
            elif result["operation"] == target_op:
                target_data = result
        
        if baseline_data and target_data:
            fw_speedup = baseline_data["fw_time"] / target_data["fw_time"]
            bw_speedup = baseline_data["bw_time"] / target_data["bw_time"]
            
            speedups_fw.append(fw_speedup)
            speedups_bw.append(bw_speedup)
            example_names.append(example.replace("_", " ").title())
    
    if not speedups_fw:
        print(f"Warning: No data found for speedup analysis between {baseline_op} and {target_op}")
        return
    
    # Create speedup plot
    x = np.arange(len(example_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, speedups_fw, width, label='Forward', alpha=0.8)
    bars2 = ax.bar(x + width/2, speedups_bw, width, label='Backward', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax.set_xlabel('Examples')
    ax.set_ylabel(f'Speedup ({baseline_op} / {target_op})')
    ax.set_title(f'Speedup Analysis: {target_op} vs {baseline_op}')
    ax.set_xticks(x)
    ax.set_xticklabels(example_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"speedup_{target_op.replace(' ', '_')}_{baseline_op.replace(' ', '_')}.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved speedup analysis: {output_path / filename}")
    plt.close()


def get_metric_label(metric: str) -> str:
    """Get human-readable label for metric."""
    labels = {
        "fw_time": "Forward Time (ms)",
        "bw_time": "Backward Time (ms)", 
        "fw_tflops": "Forward TFLOPS",
        "bw_tflops": "Backward TFLOPS"
    }
    return labels.get(metric, metric)


def create_combined_dashboard(data: Dict, output_dir: str = "benchmark_plots"):
    """Create a comprehensive dashboard with all benchmark data in one view."""
    examples = get_available_examples(data)
    if not examples:
        print("No examples found in data")
        return
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1])
    
    # Color palette for consistency
    colors = sns.color_palette("husl", n_colors=len(examples))
    operation_colors = {"causal FA2": "#1f77b4", "F.sdpa + mask": "#ff7f0e", 
                       "flexattention": "#2ca02c", "F.sdpa cross": "#d62728", 
                       "flexattention cross": "#9467bd"}
    
    # 1. Forward Time Comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    create_metric_comparison(data, examples, "fw_time", ax1, operation_colors)
    ax1.set_title("Forward Time Comparison", fontsize=14, fontweight='bold')
    
    # 2. Forward TFLOPS Comparison (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    create_metric_comparison(data, examples, "fw_tflops", ax2, operation_colors)
    ax2.set_title("Forward TFLOPS Comparison", fontsize=14, fontweight='bold')
    
    # 3. Backward Time Comparison (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    create_metric_comparison(data, examples, "bw_time", ax3, operation_colors)
    ax3.set_title("Backward Time Comparison", fontsize=14, fontweight='bold')
    
    # 4. Performance Heatmap (second row, spans all columns)
    ax4 = fig.add_subplot(gs[1, :])
    create_performance_heatmap(data, examples, ax4)
    ax4.set_title("Performance Heatmap (Forward TFLOPS)", fontsize=14, fontweight='bold')
    
    # 5. Speedup Analysis (third row, left)
    ax5 = fig.add_subplot(gs[2, 0])
    create_speedup_chart(data, examples, ax5)
    ax5.set_title("FlexAttention vs SDPA Speedup", fontsize=14, fontweight='bold')
    
    # 6. Operation Distribution (third row, middle)
    ax6 = fig.add_subplot(gs[2, 1])
    create_operation_distribution(data, examples, ax6, operation_colors)
    ax6.set_title("Performance Distribution", fontsize=14, fontweight='bold')
    
    # 7. Efficiency Chart (third row, right)
    ax7 = fig.add_subplot(gs[2, 2])
    create_efficiency_chart(data, examples, ax7)
    ax7.set_title("TFLOPS Efficiency Ratio", fontsize=14, fontweight='bold')
    
    # 8. Summary Statistics Table (bottom row)
    ax8 = fig.add_subplot(gs[3, :])
    create_summary_table(data, examples, ax8)
    ax8.set_title("Benchmark Summary Statistics", fontsize=14, fontweight='bold')
    
    # Add overall title
    fig.suptitle("Attention Mechanism Benchmark Dashboard", fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
    
    # Save the dashboard
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_path / "benchmark_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Dashboard saved to: {dashboard_path}")
    plt.close()


def create_metric_comparison(data: Dict, examples: List[str], metric: str, ax, operation_colors):
    """Create a comparison chart for a specific metric."""
    all_operations = set()
    plot_data = {}
    
    for example in examples:
        if example not in data or "latest" not in data[example]:
            continue
        results = data[example]["latest"]["results"]
        operations = [r["operation"] for r in results]
        values = [r[metric] for r in results]
        all_operations.update(operations)
        plot_data[example] = dict(zip(operations, values))
    
    operations = sorted(all_operations)
    x = np.arange(len(operations))
    width = 0.8 / len(examples) if len(examples) > 1 else 0.6
    
    for i, example in enumerate(examples):
        if example not in plot_data:
            continue
        values = [plot_data[example].get(op, 0) for op in operations]
        offset = (i - len(examples)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=example.replace("_", " ").title(), alpha=0.8)
        
        # Add value labels on bars for small datasets
        if len(examples) <= 3:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Operations', fontweight='bold')
    ax.set_ylabel(get_metric_label(metric), fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([op.replace(" ", "\n") for op in operations], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    if len(examples) > 1:
        ax.legend(fontsize=8)


def create_performance_heatmap(data: Dict, examples: List[str], ax):
    """Create a performance heatmap showing TFLOPS across examples and operations."""
    all_operations = set()
    matrix_data = []
    example_labels = []
    
    for example in examples:
        if example not in data or "latest" not in data[example]:
            continue
        results = data[example]["latest"]["results"]
        operations = [r["operation"] for r in results]
        values = [r["fw_tflops"] for r in results]
        all_operations.update(operations)
        
        example_data = dict(zip(operations, values))
        matrix_data.append(example_data)
        example_labels.append(example.replace("_", " ").title())
    
    operations = sorted(all_operations)
    
    # Create matrix
    heatmap_data = []
    for example_data in matrix_data:
        row = [example_data.get(op, 0) for op in operations]
        heatmap_data.append(row)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(operations)))
    ax.set_yticks(np.arange(len(example_labels)))
    ax.set_xticklabels([op.replace(" ", "\n") for op in operations], fontsize=8)
    ax.set_yticklabels(example_labels, fontsize=8)
    
    # Add text annotations
    for i in range(len(example_labels)):
        for j in range(len(operations)):
            if i < len(heatmap_data) and j < len(heatmap_data[i]):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TFLOPS', rotation=270, labelpad=15)


def create_speedup_chart(data: Dict, examples: List[str], ax):
    """Create a speedup analysis chart."""
    speedups = []
    example_names = []
    
    for example in examples:
        if example not in data or "latest" not in data[example]:
            continue
        results = data[example]["latest"]["results"]
        
        sdpa_time = None
        flex_time = None
        
        for result in results:
            if "sdpa" in result["operation"].lower() and "mask" in result["operation"].lower():
                sdpa_time = result["fw_time"]
            elif "flexattention" in result["operation"].lower():
                flex_time = result["fw_time"]
        
        if sdpa_time and flex_time:
            speedup = sdpa_time / flex_time
            speedups.append(speedup)
            example_names.append(example.replace("_", " ").title())
    
    if speedups:
        bars = ax.bar(example_names, speedups, color='lightcoral', alpha=0.8, edgecolor='black')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Speedup Factor', fontweight='bold')
        ax.set_xticklabels(example_names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)


def create_operation_distribution(data: Dict, examples: List[str], ax, operation_colors):
    """Create a violin plot showing performance distribution."""
    all_fw_times = []
    all_operations = []
    
    for example in examples:
        if example not in data or "latest" not in data[example]:
            continue
        results = data[example]["latest"]["results"]
        
        for result in results:
            all_fw_times.append(result["fw_time"])
            all_operations.append(result["operation"])
    
    if all_fw_times:
        # Create a simple box plot
        unique_ops = list(set(all_operations))
        data_by_op = [[] for _ in unique_ops]
        
        for time, op in zip(all_fw_times, all_operations):
            idx = unique_ops.index(op)
            data_by_op[idx].append(time)
        
        # Filter out empty lists
        valid_data = []
        valid_ops = []
        for i, op_data in enumerate(data_by_op):
            if op_data:
                valid_data.append(op_data)
                valid_ops.append(unique_ops[i])
        
        if valid_data:
            bp = ax.boxplot(valid_data, labels=[op.replace(" ", "\n") for op in valid_ops], 
                           patch_artist=True)
            
            # Color the boxes
            for patch, op in zip(bp['boxes'], valid_ops):
                color = operation_colors.get(op, 'lightblue')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    
    ax.set_ylabel('Forward Time (ms)', fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)


def create_efficiency_chart(data: Dict, examples: List[str], ax):
    """Create an efficiency chart showing TFLOPS ratio."""
    fw_efficiency = []
    bw_efficiency = []
    example_names = []
    
    for example in examples:
        if example not in data or "latest" not in data[example]:
            continue
        results = data[example]["latest"]["results"]
        
        if len(results) >= 2:
            fw_ratio = results[-1]["fw_tflops"] / results[0]["fw_tflops"] if results[0]["fw_tflops"] > 0 else 0
            bw_ratio = results[-1]["bw_tflops"] / results[0]["bw_tflops"] if results[0]["bw_tflops"] > 0 else 0
            
            fw_efficiency.append(fw_ratio)
            bw_efficiency.append(bw_ratio)
            example_names.append(example.replace("_", " ").title())
    
    if fw_efficiency:
        x = np.arange(len(example_names))
        width = 0.35
        
        ax.bar(x - width/2, fw_efficiency, width, label='Forward', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, bw_efficiency, width, label='Backward', alpha=0.8, color='lightcoral')
        
        ax.set_ylabel('Efficiency Ratio', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(example_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)


def create_summary_table(data: Dict, examples: List[str], ax):
    """Create a summary statistics table."""
    table_data = []
    
    for example in examples:
        if example not in data or "latest" not in data[example]:
            continue
        results = data[example]["latest"]["results"]
        
        # Calculate statistics
        fw_times = [r["fw_time"] for r in results]
        fw_tflops = [r["fw_tflops"] for r in results]
        
        best_fw_time = min(fw_times)
        best_fw_tflops = max(fw_tflops)
        best_operation = results[fw_tflops.index(best_fw_tflops)]["operation"]
        
        table_data.append([
            example.replace("_", " ").title(),
            f"{best_fw_time:.2f} ms",
            f"{best_fw_tflops:.2f}",
            best_operation,
            len(results)
        ])
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=['Example', 'Best FW Time', 'Best FW TFLOPS', 'Best Operation', 'Num Operations'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')


def create_comprehensive_report(data: Dict, output_dir: str = "benchmark_plots"):
    """Create a comprehensive visualization report with all available analyses."""
    print("Creating comprehensive benchmark visualization report...")
    
    examples = get_available_examples(data)
    if not examples:
        print("No examples found in data")
        return
    
    print(f"Found examples: {examples}")
    
    # Create the combined dashboard first
    create_combined_dashboard(data, output_dir)
    
    # Create individual comparison plots for all metrics
    for metric in ["fw_time", "bw_time", "fw_tflops", "bw_tflops"]:
        create_comparison_plot(data, examples, metric, output_dir)
    
    # Create performance matrix
    create_performance_matrix(data, examples, output_dir)
    
    # Create speedup analysis if both operations exist
    sample_example = examples[0]
    if "latest" in data[sample_example]:
        operations = [r["operation"] for r in data[sample_example]["latest"]["results"]]
        if "F.sdpa + mask" in operations and "flexattention" in operations:
            create_speedup_analysis(data, examples, "F.sdpa + mask", "flexattention", output_dir)
    
    # Create time series for each example and operation (if multiple runs exist)
    for example in examples:
        runs = data[example].get("runs", {})
        if len(runs) > 1:  # Only create time series if multiple runs exist
            if "latest" in data[example]:
                operations = [r["operation"] for r in data[example]["latest"]["results"]]
                for operation in operations:
                    create_time_series_plot(data, example, operation, output_dir)
    
    print(f"Comprehensive report generated in: {Path(output_dir).resolve()}")


def main():
    """Main function to run the visualization script."""
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    
    parser = ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_results.json",
        help="Path to the benchmark results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_plots",
        help="Directory to save the plots"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fw_tflops", "bw_tflops"],
        help="Metrics to plot (fw_tflops, bw_tflops, fw_time, bw_time)"
    )
    parser.add_argument(
        "--example",
        type=str,
        help="Specific example to plot. If not provided, all examples will be plotted."
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Create a combined dashboard with all visualizations"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Create a comprehensive report with all available visualizations"
    )
    
    args = parser.parse_args()
    
    # Load the benchmark data
    data = load_benchmark_data(args.input)
    
    if args.comprehensive:
        create_comprehensive_report(data, args.output_dir)
    elif args.dashboard:
        create_combined_dashboard(data, args.output_dir)
    elif args.example:
        if args.example not in data:
            print(f"Example {args.example} not found in benchmark data")
            print(f"Available examples: {', '.join(data.keys())}")
            return
        
        for metric in args.metrics:
            plot_benchmark_results(
                data,
                args.example,
                metric,
                save_path=str(Path(args.output_dir) / f"{args.example}_{metric}.png")
            )
    else:
        plot_all_examples(data, args.metrics, args.output_dir)


if __name__ == "__main__":
    main() 