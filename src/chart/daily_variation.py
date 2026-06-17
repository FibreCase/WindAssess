"""
Daily variation plotting for wind speed data.

Generates daily average wind speed variation plots for both radar and tower
data across all measurement heights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_file import import_data


# Tower and radar height configurations
TOWER_HEIGHTS = [2, 5, 10, 20, 50, 80]
RADAR_HEIGHTS = list(range(40, 201, 5))

# Output directory
OUTPUT_DIR = "result/chart/daily_variation"


def get_tower_speed_column(height):
    """Get the wind speed column name for tower data."""
    return f"Avg Wind Speed @ {height}m [m/s]"


def get_radar_speed_column(height):
    """Get the wind speed column name for radar data."""
    return f"Wind Speed{height}m"


def compute_daily_variation(df, time_col, heights, get_column_fn):
    """
    Compute daily average wind speed variation for all heights.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime column and wind speed data.
    time_col : str
        Name of the datetime column.
    heights : list[int]
        List of heights to process.
    get_column_fn : callable
        Function that returns column name for a given height.

    Returns
    -------
    dict
        Dictionary mapping height to DataFrame with columns:
        'hour', 'mean', 'std', 'min', 'max'.
    """
    results = {}

    for h in heights:
        col = get_column_fn(h)

        if col not in df.columns:
            continue

        # Extract hour and group
        df_copy = df[[time_col, col]].copy()
        df_copy['hour'] = df_copy[time_col].dt.hour
        df_copy = df_copy.dropna(subset=[col])

        if len(df_copy) == 0:
            continue

        # Compute statistics by hour
        hourly_stats = df_copy.groupby('hour')[col].agg(['mean', 'std', 'min', 'max'])
        hourly_stats = hourly_stats.reset_index()

        results[h] = hourly_stats

    return results


def plot_daily_variation_individual(stats_dict, heights, title, output_path):
    """
    Plot daily variation with individual subplots for each height.

    Parameters
    ----------
    stats_dict : dict
        Dictionary mapping height to hourly statistics DataFrame.
    heights : list[int]
        List of heights to plot.
    title : str
        Plot title.
    output_path : str
        Path to save the output figure.
    """
    n_heights = len(heights)
    n_cols = 3
    n_rows = (n_heights + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_heights))

    for idx, h in enumerate(heights):
        if h not in stats_dict:
            axes[idx].text(0.5, 0.5, 'No data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{h}m')
            continue

        stats = stats_dict[h]
        hours = stats['hour']

        # Plot mean line
        axes[idx].plot(hours, stats['mean'], color=colors[idx], linewidth=2, label='Mean')

        # Plot shaded area for min-max range
        axes[idx].fill_between(hours, stats['min'], stats['max'],
                              alpha=0.2, color=colors[idx], label='Min-Max range')

        # Plot std deviation band
        axes[idx].fill_between(hours,
                              stats['mean'] - stats['std'],
                              stats['mean'] + stats['std'],
                              alpha=0.2, color=colors[idx], label='Std dev range')

        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Wind Speed (m/s)')
        axes[idx].set_title(f'{h}m')
        axes[idx].set_xticks(range(0, 24, 3))
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_heights, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_daily_variation_combined(stats_dict, heights, title, output_path):
    """
    Plot all daily variations on a single figure.

    Parameters
    ----------
    stats_dict : dict
        Dictionary mapping height to hourly statistics DataFrame.
    heights : list[int]
        List of heights to plot.
    title : str
        Plot title.
    output_path : str
        Path to save the output figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate colors based on height
    norm_heights = [(h - min(heights)) / (max(heights) - min(heights)) for h in heights]
    colors = plt.cm.viridis(norm_heights)

    for h, color in zip(heights, colors):
        if h not in stats_dict:
            continue

        stats = stats_dict[h]
        hours = stats['hour']

        # Plot mean line
        ax.plot(hours, stats['mean'], linewidth=2, color=color, label=f'{h}m')

        # Plot shaded area for std deviation
        ax.fill_between(hours,
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       alpha=0.15, color=color)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Height', fontsize=8, title_fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Generate daily variation plots for radar and tower data."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import data
    print("Loading radar data...")
    radar_df = import_data("data/radar.csv")

    print("Loading tower data...")
    tower_df = import_data("data/tower.csv")

    # Determine time column names
    radar_time_col = "Time" if "Time" in radar_df.columns else "timestamp"
    tower_time_col = "Time" if "Time" in tower_df.columns else "timestamp"

    # Compute daily variation for radar
    print("\nComputing radar daily variation...")
    radar_stats = compute_daily_variation(
        df=radar_df,
        time_col=radar_time_col,
        heights=RADAR_HEIGHTS,
        get_column_fn=get_radar_speed_column
    )

    # Compute daily variation for tower
    print("Computing tower daily variation...")
    tower_stats = compute_daily_variation(
        df=tower_df,
        time_col=tower_time_col,
        heights=TOWER_HEIGHTS,
        get_column_fn=get_tower_speed_column
    )

    # Plot radar daily variation (individual subplots)
    print("\nGenerating radar daily variation plot...")
    plot_daily_variation_individual(
        stats_dict=radar_stats,
        heights=RADAR_HEIGHTS,
        title="Radar Wind Speed Daily Variation",
        output_path=os.path.join(OUTPUT_DIR, "radar_daily_variation.png")
    )

    # Plot radar combined daily variation
    print("Generating radar combined daily variation plot...")
    plot_daily_variation_combined(
        stats_dict=radar_stats,
        heights=RADAR_HEIGHTS,
        title="Radar Wind Speed Daily Variation (All Heights)",
        output_path=os.path.join(OUTPUT_DIR, "radar_daily_variation_combined.png")
    )

    # Plot tower daily variation (individual subplots)
    print("\nGenerating tower daily variation plot...")
    plot_daily_variation_individual(
        stats_dict=tower_stats,
        heights=TOWER_HEIGHTS,
        title="Tower Wind Speed Daily Variation",
        output_path=os.path.join(OUTPUT_DIR, "tower_daily_variation.png")
    )

    # Plot tower combined daily variation
    print("Generating tower combined daily variation plot...")
    plot_daily_variation_combined(
        stats_dict=tower_stats,
        heights=TOWER_HEIGHTS,
        title="Tower Wind Speed Daily Variation (All Heights)",
        output_path=os.path.join(OUTPUT_DIR, "tower_daily_variation_combined.png")
    )

    print("\nDone! All daily variation plots generated.")


if __name__ == "__main__":
    main()
