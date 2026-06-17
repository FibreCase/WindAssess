"""
Wind shear daily variation plotting.

Generates daily wind shear exponent variation plots for tower data.
Wind shear is calculated using the power law between adjacent height pairs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_file import import_data


# Tower heights
TOWER_HEIGHTS = [2, 5, 10, 20, 50, 80]

# Output directory
OUTPUT_DIR = "result/chart/shear_variation"


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
    # Avoid division by zero and log of negative numbers
    mask = (ws1 > 0) & (ws2 > 0)

    alpha = np.full_like(ws1, np.nan, dtype=float)
    alpha[mask] = np.log(ws2[mask] / ws1[mask]) / np.log(h2 / h1)

    return alpha


def compute_shear_daily_variation(df, time_col, heights):
    """
    Compute daily wind shear variation for adjacent height pairs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime and wind speed columns.
    time_col : str
        Name of the datetime column.
    heights : list[int]
        List of measurement heights.

    Returns
    -------
    dict
        Dictionary mapping height pair string to DataFrame with hourly statistics.
    """
    results = {}

    # Calculate shear for consecutive height pairs
    for i in range(len(heights) - 1):
        h1 = heights[i]
        h2 = heights[i + 1]

        ws1_col = get_tower_speed_column(h1)
        ws2_col = get_tower_speed_column(h2)

        if ws1_col not in df.columns or ws2_col not in df.columns:
            continue

        # Calculate shear exponent
        df_copy = df[[time_col, ws1_col, ws2_col]].copy()
        df_copy = df_copy.dropna(subset=[ws1_col, ws2_col])

        if len(df_copy) == 0:
            continue

        df_copy['shear'] = calculate_shear_exponent(
            df_copy[ws1_col].values, h1,
            df_copy[ws2_col].values, h2
        )

        # Remove invalid shear values
        df_copy = df_copy.dropna(subset=['shear'])

        if len(df_copy) == 0:
            continue

        # Extract hour and compute statistics
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


def plot_shear_individual(results, title, output_path):
    """
    Plot daily shear variation with individual subplots for each height pair.
    """
    n_pairs = len(results)

    if n_pairs == 0:
        print("No shear data to plot.")
        return

    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_pairs))

    for idx, (pair_key, data) in enumerate(results.items()):
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
        axes[idx].set_ylabel('Shear Exponent (α)')
        axes[idx].set_title(f'{h1}-{h2}m')
        axes[idx].set_xticks(range(0, 24, 3))
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=6)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_shear_combined(results, title, output_path):
    """
    Plot all shear variations on a single figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    n_pairs = len(results)
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_pairs))

    for idx, (pair_key, data) in enumerate(results.items()):
        stats = data['stats']
        h1 = data['h1']
        h2 = data['h2']
        hours = stats['hour']
        color = colors[idx]

        ax.plot(hours, stats['mean'], linewidth=2, color=color,
               label=f'{h1}-{h2}m')
        ax.fill_between(hours,
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       alpha=0.15, color=color)

    # Reference line for typical shear exponent
    ax.axhline(y=0.143, color='red', linestyle='--', linewidth=1.5,
              label='Typical (0.143)', alpha=0.7)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Shear Exponent (α)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Height Pair', fontsize=8, title_fontsize=9,
             loc='upper right', framealpha=0.9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Generate wind shear daily variation plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tower data...")
    tower_df = import_data("data/tower.csv")

    tower_time_col = "Time" if "Time" in tower_df.columns else "timestamp"

    print("Computing wind shear daily variation...")
    shear_results = compute_shear_daily_variation(
        df=tower_df,
        time_col=tower_time_col,
        heights=TOWER_HEIGHTS
    )

    if not shear_results:
        print("No shear data available. Skipping shear plots.")
        return

    print("\nGenerating wind shear daily variation plot...")
    plot_shear_individual(
        results=shear_results,
        title="Wind Shear Exponent Daily Variation",
        output_path=os.path.join(OUTPUT_DIR, "shear_daily_variation.png")
    )

    print("Generating wind shear combined plot...")
    plot_shear_combined(
        results=shear_results,
        title="Wind Shear Exponent Daily Variation (All Height Pairs)",
        output_path=os.path.join(OUTPUT_DIR, "shear_daily_variation_combined.png")
    )

    print("\nDone! Wind shear plots generated.")


if __name__ == "__main__":
    main()
