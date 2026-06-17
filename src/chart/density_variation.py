"""
Air density daily variation plotting.

Generates daily air density variation plots from tower meteorological data.
Air density is calculated using temperature and pressure measurements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_file import import_data


# Tower heights with temperature measurements
TOWER_TEMP_HEIGHTS = [2, 50, 80]


def calculate_air_density(temp_c, pressure_mbar):
    """
    Calculate air density from temperature and pressure.

    Uses the ideal gas law: rho = P / (R_specific * T)

    Parameters
    ----------
    temp_c : float or array-like
        Temperature in degrees Celsius.
    pressure_mbar : float or array-like
        Pressure in millibars (hPa).

    Returns
    -------
    float or array-like
        Air density in kg/m^3.
    """
    # Convert to SI units
    temp_k = temp_c + 273.15  # Celsius to Kelvin
    pressure_pa = pressure_mbar * 100  # mbar to Pa

    # Specific gas constant for dry air
    R_specific = 287.058  # J/(kg*K)

    # Calculate density
    density = pressure_pa / (R_specific * temp_k)

    return density


def compute_density_daily_variation(df, time_col):
    """
    Compute daily air density variation for all temperature measurement heights.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime, temperature, and pressure columns.
    time_col : str
        Name of the datetime column.

    Returns
    -------
    dict
        Dictionary mapping height to DataFrame with hourly statistics.
    """
    results = {}

    # Get pressure column (station pressure is typically at one height)
    pressure_col = "Station Pressure [mBar]"

    if pressure_col not in df.columns:
        print(f"Warning: Pressure column '{pressure_col}' not found.")
        return results

    for h in TOWER_TEMP_HEIGHTS:
        temp_col = f"Temperature @ {h}m [deg C]"

        if temp_col not in df.columns:
            continue

        # Calculate air density
        df_copy = df[[time_col, temp_col, pressure_col]].copy()
        df_copy = df_copy.dropna(subset=[temp_col, pressure_col])

        if len(df_copy) == 0:
            continue

        # Calculate density
        df_copy['density'] = calculate_air_density(
            df_copy[temp_col],
            df_copy[pressure_col]
        )

        # Extract hour and compute statistics
        df_copy['hour'] = df_copy[time_col].dt.hour
        hourly_stats = df_copy.groupby('hour')['density'].agg(['mean', 'std', 'min', 'max'])
        hourly_stats = hourly_stats.reset_index()

        results[h] = hourly_stats

    return results


def plot_density_individual(stats_dict, heights, title, output_path):
    """
    Plot daily density variation with individual subplots for each height.
    """
    n_heights = len(heights)
    n_cols = 2
    n_rows = (n_heights + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_heights))

    for idx, h in enumerate(heights):
        if h not in stats_dict:
            axes[idx].text(0.5, 0.5, 'No data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{h}m')
            continue

        stats = stats_dict[h]
        hours = stats['hour']

        axes[idx].plot(hours, stats['mean'], color=colors[idx], linewidth=2, label='Mean')
        axes[idx].fill_between(hours, stats['min'], stats['max'],
                              alpha=0.2, color=colors[idx], label='Min-Max range')
        axes[idx].fill_between(hours,
                              stats['mean'] - stats['std'],
                              stats['mean'] + stats['std'],
                              alpha=0.2, color=colors[idx], label='Std dev range')

        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Air Density (kg/m³)')
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


def plot_density_combined(stats_dict, heights, title, output_path):
    """
    Plot all density variations on a single figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    norm_heights = [(h - min(heights)) / (max(heights) - min(heights)) for h in heights]
    colors = plt.cm.plasma(norm_heights)

    for h, color in zip(heights, colors):
        if h not in stats_dict:
            continue

        stats = stats_dict[h]
        hours = stats['hour']

        ax.plot(hours, stats['mean'], linewidth=2, color=color, label=f'{h}m')
        ax.fill_between(hours,
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       alpha=0.15, color=color)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Air Density (kg/m³)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Height', fontsize=9, title_fontsize=10, loc='upper right', framealpha=0.9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Generate air density daily variation plots."""
    os.makedirs("result/chart/tower/density_variation", exist_ok=True)

    print("Loading tower data...")
    tower_df = import_data("data/tower.csv")

    tower_time_col = "Time" if "Time" in tower_df.columns else "timestamp"

    print("Computing air density daily variation...")
    density_stats = compute_density_daily_variation(
        df=tower_df,
        time_col=tower_time_col
    )

    if not density_stats:
        print("No density data available. Skipping density plots.")
        return

    print("\nGenerating air density daily variation plot...")
    plot_density_individual(
        stats_dict=density_stats,
        heights=TOWER_TEMP_HEIGHTS,
        title="Air Density Daily Variation",
        output_path="result/chart/tower/density_variation/density_daily_variation.png"
    )

    print("Generating air density combined plot...")
    plot_density_combined(
        stats_dict=density_stats,
        heights=TOWER_TEMP_HEIGHTS,
        title="Air Density Daily Variation (All Heights)",
        output_path="result/chart/tower/density_variation/density_daily_variation_combined.png"
    )

    print("\nDone! Air density plots generated.")


if __name__ == "__main__":
    main()
