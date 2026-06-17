"""
Weibull distribution plotting for wind speed data.

Generates Weibull distribution plots for both radar and tower wind speed data
across all measurement heights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from src.data_file import import_data


# Tower and radar height configurations
TOWER_HEIGHTS = [2, 5, 10, 20, 50, 80]
RADAR_HEIGHTS = list(range(40, 201, 5))


def get_tower_speed_column(height):
    """Get the wind speed column name for tower data."""
    return f"Avg Wind Speed @ {height}m [m/s]"


def get_radar_speed_column(height):
    """Get the wind speed column name for radar data."""
    return f"Wind Speed{height}m"


def fit_weibull(data):
    """
    Fit Weibull distribution to wind speed data.

    Parameters
    ----------
    data : array-like
        Wind speed data (must be non-negative).

    Returns
    -------
    tuple
        (shape, loc, scale) parameters of the fitted Weibull distribution.
    """
    # Filter out NaN and negative values
    clean_data = data[~np.isnan(data) & (data >= 0)]

    if len(clean_data) < 10:
        return None, None, None

    # Fit Weibull distribution
    shape, loc, scale = weibull_min.fit(clean_data, floc=0)

    return shape, loc, scale


def plot_weibull_distribution(df, heights, get_column_fn, title, output_path):
    """
    Plot Weibull distributions for all heights.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wind speed data.
    heights : list[int]
        List of heights to plot.
    get_column_fn : callable
        Function that returns column name for a given height.
    title : str
        Plot title.
    output_path : str
        Path to save the output figure.
    """
    # Use a grid layout - adjust based on number of heights
    n_heights = len(heights)
    n_cols = 3
    n_rows = (n_heights + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_heights))

    for idx, h in enumerate(heights):
        col = get_column_fn(h)

        if col not in df.columns:
            continue

        # Get clean data
        data = df[col].dropna()
        data = data[data >= 0]

        if len(data) < 10:
            axes[idx].text(0.5, 0.5, 'Insufficient data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{h}m')
            continue

        # Fit Weibull distribution
        shape, loc, scale = fit_weibull(df[col].values)

        if shape is None:
            axes[idx].text(0.5, 0.5, 'Fit failed',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{h}m')
            continue

        # Plot histogram
        axes[idx].hist(data, bins=30, density=True, alpha=0.6,
                      color=colors[idx], label='Data', edgecolor='black', linewidth=0.5)

        # Plot fitted Weibull PDF
        x = np.linspace(0, max(data.max() * 1.1, 1), 200)
        pdf = weibull_min.pdf(x, shape, loc, scale)
        axes[idx].plot(x, pdf, 'r-', linewidth=2,
                      label=f'Weibull\nk={shape:.2f}\nc={scale:.2f}')

        axes[idx].set_xlabel('Wind Speed (m/s)')
        axes[idx].set_ylabel('Probability Density')
        axes[idx].set_title(f'{h}m')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_heights, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_weibull_combined(df, heights, get_column_fn, title, output_path):
    """
    Plot all Weibull distributions on a single figure.

    Creates a combined plot showing the Weibull PDF curves for all heights
    on the same axes, colored by height.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wind speed data.
    heights : list[int]
        List of heights to plot.
    get_column_fn : callable
        Function that returns column name for a given height.
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
        col = get_column_fn(h)

        if col not in df.columns:
            continue

        # Get clean data
        data = df[col].dropna()
        data = data[data >= 0]

        if len(data) < 10:
            continue

        # Fit Weibull distribution
        shape, loc, scale = fit_weibull(df[col].values)

        if shape is None:
            continue

        # Plot fitted Weibull PDF
        x = np.linspace(0, max(data.max() * 1.1, 1), 200)
        pdf = weibull_min.pdf(x, shape, loc, scale)
        ax.plot(x, pdf, linewidth=2, color=color, label=f'{h}m (k={shape:.2f}, c={scale:.2f})')

    ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Height', fontsize=8, title_fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Generate Weibull distribution plots for radar and tower data."""
    # Create output directories
    os.makedirs("result/chart/radar/weibull", exist_ok=True)
    os.makedirs("result/chart/tower/weibull", exist_ok=True)

    # Import data
    print("Loading radar data...")
    radar_df = import_data("data/radar.csv")

    print("Loading tower data...")
    tower_df = import_data("data/tower.csv")

    # Plot radar Weibull distributions (individual subplots)
    print("\nGenerating radar Weibull distribution plot...")
    plot_weibull_distribution(
        df=radar_df,
        heights=RADAR_HEIGHTS,
        get_column_fn=get_radar_speed_column,
        title="Radar Wind Speed Weibull Distribution",
        output_path="result/chart/radar/weibull/radar_weibull.png"
    )

    # Plot radar combined Weibull distributions
    print("Generating radar combined Weibull distribution plot...")
    plot_weibull_combined(
        df=radar_df,
        heights=RADAR_HEIGHTS,
        get_column_fn=get_radar_speed_column,
        title="Radar Wind Speed Weibull Distribution (All Heights)",
        output_path="result/chart/radar/weibull/radar_weibull_combined.png"
    )

    # Plot tower Weibull distributions (individual subplots)
    print("\nGenerating tower Weibull distribution plot...")
    plot_weibull_distribution(
        df=tower_df,
        heights=TOWER_HEIGHTS,
        get_column_fn=get_tower_speed_column,
        title="Tower Wind Speed Weibull Distribution",
        output_path="result/chart/tower/weibull/tower_weibull.png"
    )

    # Plot tower combined Weibull distributions
    print("Generating tower combined Weibull distribution plot...")
    plot_weibull_combined(
        df=tower_df,
        heights=TOWER_HEIGHTS,
        get_column_fn=get_tower_speed_column,
        title="Tower Wind Speed Weibull Distribution (All Heights)",
        output_path="result/chart/tower/weibull/tower_weibull_combined.png"
    )

    print("\nDone! All plots generated.")


if __name__ == "__main__":
    main()
