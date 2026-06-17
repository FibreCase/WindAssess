"""
Wind shear daily variation plotting.

Generates daily wind shear exponent variation plots.
Supports both radar (multi-height) and tower data formats.
Wind shear is calculated using the power law between adjacent height pairs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_file import import_data


# Radar heights
RADAR_HEIGHTS = list(range(40, 201, 5))

# Tower heights
TOWER_HEIGHTS = [2, 5, 10, 20, 50, 80]

# Output directories
RADAR_OUTPUT_DIR = "result/chart/radar/shear_variation"
TOWER_OUTPUT_DIR = "result/chart/tower/shear_variation"


def get_radar_speed_column(height):
    """Get the wind speed column name for radar data."""
    return f"Wind Speed{height}m"


def get_tower_speed_column(height):
    """Get the wind speed column name for tower data."""
    return f"Avg Wind Speed @ {height}m [m/s]"


def calculate_shear_exponent(ws1, h1, ws2, h2):
    """
    Calculate wind shear exponent using the power law.

    The power law is: V2/V1 = (h2/h1)^alpha
    Solving for alpha: alpha = ln(V2/V1) / ln(h2/h1)

    Parameters
    ----------
    ws1 : float or array-like
        Wind speed at lower height (m/s).
    h1 : float
        Lower height (m).
    ws2 : float or array-like
        Wind speed at upper height (m/s).
    h2 : float
        Upper height (m).

    Returns
    -------
    float or array-like
        Shear exponent (alpha).
    """
    mask = (ws1 > 0) & (ws2 > 0)
    alpha = np.full_like(ws1, np.nan, dtype=float)
    alpha[mask] = np.log(ws2[mask] / ws1[mask]) / np.log(h2 / h1)
    return alpha


def compute_shear_daily_variation(df, time_col, heights, get_column_fn):
    """
    Compute daily wind shear variation for adjacent height pairs.
    """
    results = {}

    for i in range(len(heights) - 1):
        h1 = heights[i]
        h2 = heights[i + 1]

        ws1_col = get_column_fn(h1)
        ws2_col = get_column_fn(h2)

        if ws1_col not in df.columns or ws2_col not in df.columns:
            continue

        df_copy = df[[time_col, ws1_col, ws2_col]].copy()
        df_copy = df_copy.dropna(subset=[ws1_col, ws2_col])

        if len(df_copy) == 0:
            continue

        df_copy['shear'] = calculate_shear_exponent(
            df_copy[ws1_col].values, h1,
            df_copy[ws2_col].values, h2
        )

        df_copy = df_copy.dropna(subset=['shear'])

        if len(df_copy) == 0:
            continue

        df_copy['hour'] = df_copy[time_col].dt.hour
        hourly_stats = df_copy.groupby('hour')['shear'].agg(['mean', 'std', 'min', 'max'])
        hourly_stats = hourly_stats.reset_index()

        pair_key = f"{h1}-{h2}m"
        results[pair_key] = {
            'stats': hourly_stats,
            'h1': h1,
            'h2': h2
        }

    return results


def plot_shear_individual(results, title, output_path, max_plots=6):
    """Plot daily shear variation with individual subplots."""
    n_pairs = min(len(results), max_plots)

    if n_pairs == 0:
        print("No shear data to plot.")
        return

    keys = list(results.keys())
    if len(results) > max_plots:
        step = len(keys) // max_plots
        selected_keys = keys[::step][:max_plots]
    else:
        selected_keys = keys

    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_pairs))

    for idx, pair_key in enumerate(selected_keys):
        data = results[pair_key]
        stats = data['stats']
        h1 = data['h1']
        h2 = data['h2']
        hours = stats['hour']

        axes[idx].plot(hours, stats['mean'], color=colors[idx], linewidth=2, label='Mean')
        axes[idx].fill_between(hours, stats['min'], stats['max'],
                              alpha=0.2, color=colors[idx], label='Min-Max range')
        axes[idx].fill_between(hours,
                              stats['mean'] - stats['std'],
                              stats['mean'] + stats['std'],
                              alpha=0.2, color=colors[idx], label='Std dev range')
        axes[idx].axhline(y=0.143, color='red', linestyle='--', linewidth=1,
                         label='Typical (0.143)', alpha=0.7)

        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Shear Exponent (alpha)')
        axes[idx].set_title(f'{h1}-{h2}m')
        axes[idx].set_xticks(range(0, 24, 3))
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=6)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_shear_combined(results, title, output_path, max_plots=8):
    """Plot all shear variations on a single figure."""
    fig, ax = plt.subplots(figsize=(12, 8))

    keys = list(results.keys())
    if len(keys) > max_plots:
        step = len(keys) // max_plots
        selected_keys = keys[::step][:max_plots]
    else:
        selected_keys = keys

    n_pairs = len(selected_keys)
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_pairs))

    for idx, pair_key in enumerate(selected_keys):
        data = results[pair_key]
        stats = data['stats']
        h1 = data['h1']
        h2 = data['h2']
        hours = stats['hour']
        color = colors[idx]

        ax.plot(hours, stats['mean'], linewidth=2, color=color, label=f'{h1}-{h2}m')
        ax.fill_between(hours,
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       alpha=0.15, color=color)

    ax.axhline(y=0.143, color='red', linestyle='--', linewidth=1.5,
              label='Typical (0.143)', alpha=0.7)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Shear Exponent (alpha)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Height Pair', fontsize=7, title_fontsize=8,
             loc='upper right', framealpha=0.9, ncol=2)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Generate wind shear daily variation plots for radar and tower data."""
    os.makedirs(RADAR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOWER_OUTPUT_DIR, exist_ok=True)

    # Import radar data
    print("Loading radar data...")
    radar_df = import_data("data/radar.csv")
    radar_time_col = "Time" if "Time" in radar_df.columns else "timestamp"

    print("Computing radar wind shear daily variation...")
    radar_shear_results = compute_shear_daily_variation(
        df=radar_df,
        time_col=radar_time_col,
        heights=RADAR_HEIGHTS,
        get_column_fn=get_radar_speed_column
    )

    if radar_shear_results:
        print("\nGenerating radar wind shear daily variation plot...")
        plot_shear_individual(
            results=radar_shear_results,
            title="Radar Wind Shear Exponent Daily Variation",
            output_path=os.path.join(RADAR_OUTPUT_DIR, "shear_daily_variation.png"),
            max_plots=6
        )

        print("Generating radar wind shear combined plot...")
        plot_shear_combined(
            results=radar_shear_results,
            title="Radar Wind Shear Exponent Daily Variation (Selected Height Pairs)",
            output_path=os.path.join(RADAR_OUTPUT_DIR, "shear_daily_variation_combined.png"),
            max_plots=8
        )
    else:
        print("\nNo radar shear data available.")

    # Import tower data
    print("\nLoading tower data...")
    tower_df = import_data("data/tower.csv")
    tower_time_col = "Time" if "Time" in tower_df.columns else "timestamp"

    print("Computing tower wind shear daily variation...")
    tower_shear_results = compute_shear_daily_variation(
        df=tower_df,
        time_col=tower_time_col,
        heights=TOWER_HEIGHTS,
        get_column_fn=get_tower_speed_column
    )

    if tower_shear_results:
        print("\nGenerating tower wind shear daily variation plot...")
        plot_shear_individual(
            results=tower_shear_results,
            title="Tower Wind Shear Exponent Daily Variation",
            output_path=os.path.join(TOWER_OUTPUT_DIR, "shear_daily_variation.png"),
            max_plots=6
        )

        print("Generating tower wind shear combined plot...")
        plot_shear_combined(
            results=tower_shear_results,
            title="Tower Wind Shear Exponent Daily Variation (All Height Pairs)",
            output_path=os.path.join(TOWER_OUTPUT_DIR, "shear_daily_variation_combined.png"),
            max_plots=8
        )
    else:
        print("\nNo tower shear data available.")

    print("\nDone! Wind shear plots generated.")


if __name__ == "__main__":
    main()
