"""
Air density daily variation plotting.

Generates daily air density variation plots from meteorological data.
Supports both radar (single station) and tower (multi-height) data formats.
Air density is calculated using temperature and pressure measurements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_file import import_data


# Output directory
RADAR_OUTPUT_DIR = "result/chart/radar/density_variation"
TOWER_OUTPUT_DIR = "result/chart/tower/density_variation"

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


def get_pressure_column(df):
    """
    Detect the pressure column name from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    str or None
        Pressure column name, or None if not found.
    """
    possible_names = ["Station Pressure [mBar]", "Pressure"]
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def get_temperature_columns(df):
    """
    Detect temperature column(s) from DataFrame.

    Returns a list of (height, column_name) tuples.
    For radar data: returns [(None, "Temperature")]
    For tower data: returns [(2, "Temperature @ 2m [deg C]"), ...]

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list
        List of (height, column_name) tuples.
    """
    results = []

    # Check for tower-style columns
    for h in TOWER_TEMP_HEIGHTS:
        col = f"Temperature @ {h}m [deg C]"
        if col in df.columns:
            results.append((h, col))

    # If no tower columns found, check for radar-style single column
    if not results:
        if "Temperature" in df.columns:
            results.append((None, "Temperature"))

    return results


def compute_density_daily_variation(df, time_col):
    """
    Compute daily air density variation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime, temperature, and pressure columns.
    time_col : str
        Name of the datetime column.

    Returns
    -------
    dict
        Dictionary mapping height (or 'station') to DataFrame with hourly statistics.
    """
    results = {}

    pressure_col = get_pressure_column(df)
    if pressure_col is None:
        print("Warning: No pressure column found.")
        return results

    temp_columns = get_temperature_columns(df)
    if not temp_columns:
        print("Warning: No temperature columns found.")
        return results

    for h, temp_col in temp_columns:
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

        # Use height or 'station' as key
        key = f"{h}m" if h is not None else "station"
        results[key] = hourly_stats

    return results


def plot_density_individual(stats_dict, title, output_path, is_tower=True):
    """
    Plot daily density variation with individual subplots.
    """
    n_plots = len(stats_dict)

    if n_plots == 0:
        print("No density data to plot.")
        return

    if is_tower:
        n_cols = 2
        keys = list(stats_dict.keys())
    else:
        n_cols = 1
        keys = list(stats_dict.keys())

    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, max(n_plots, 1)))

    for idx, key in enumerate(keys):
        stats = stats_dict[key]
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
        axes[idx].set_title(f'Height: {key}' if key != 'station' else 'Station')
        axes[idx].set_xticks(range(0, 24, 3))
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_density_combined(stats_dict, title, output_path):
    """
    Plot all density variations on a single figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    n_plots = len(stats_dict)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, max(n_plots, 1)))

    for idx, (key, stats) in enumerate(stats_dict.items()):
        hours = stats['hour']
        color = colors[idx]

        label = key if key != 'station' else 'Station'
        ax.plot(hours, stats['mean'], linewidth=2, color=color, label=label)
        ax.fill_between(hours,
                       stats['mean'] - stats['std'],
                       stats['mean'] + stats['std'],
                       alpha=0.15, color=color)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Air Density (kg/m³)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Location', fontsize=9, title_fontsize=10, loc='upper right', framealpha=0.9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Generate air density daily variation plots for radar and tower data."""
    os.makedirs(RADAR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOWER_OUTPUT_DIR, exist_ok=True)

    # Import radar data
    print("Loading radar data...")
    radar_df = import_data("data/radar.csv")
    radar_time_col = "Time" if "Time" in radar_df.columns else "timestamp"

    print("Computing radar air density daily variation...")
    radar_density_stats = compute_density_daily_variation(
        df=radar_df,
        time_col=radar_time_col
    )

    if radar_density_stats:
        print("\nGenerating radar air density daily variation plot...")
        plot_density_individual(
            stats_dict=radar_density_stats,
            title="Radar Air Density Daily Variation",
            output_path=os.path.join(RADAR_OUTPUT_DIR, "density_daily_variation.png"),
            is_tower=False
        )

        print("Generating radar air density combined plot...")
        plot_density_combined(
            stats_dict=radar_density_stats,
            title="Radar Air Density Daily Variation",
            output_path=os.path.join(RADAR_OUTPUT_DIR, "density_daily_variation_combined.png")
        )
    else:
        print("\nNo radar density data available.")

    # Import tower data
    print("\nLoading tower data...")
    tower_df = import_data("data/tower.csv")
    tower_time_col = "Time" if "Time" in tower_df.columns else "timestamp"

    print("Computing tower air density daily variation...")
    tower_density_stats = compute_density_daily_variation(
        df=tower_df,
        time_col=tower_time_col
    )

    if tower_density_stats:
        print("\nGenerating tower air density daily variation plot...")
        plot_density_individual(
            stats_dict=tower_density_stats,
            title="Tower Air Density Daily Variation",
            output_path=os.path.join(TOWER_OUTPUT_DIR, "density_daily_variation.png"),
            is_tower=True
        )

        print("Generating tower air density combined plot...")
        plot_density_combined(
            stats_dict=tower_density_stats,
            title="Tower Air Density Daily Variation (All Heights)",
            output_path=os.path.join(TOWER_OUTPUT_DIR, "density_daily_variation_combined.png")
        )
    else:
        print("\nNo tower density data available.")

    print("\nDone! Air density plots generated.")


if __name__ == "__main__":
    main()
