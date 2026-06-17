"""
Wind rose and energy rose plotting.

Generates wind direction rose plots and wind energy rose plots for both
radar and tower data across measurement heights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_file import import_data


# Radar and tower height configurations
RADAR_HEIGHTS = list(range(40, 201, 5))
TOWER_HEIGHTS = [2, 5, 10, 20, 50, 80]

# Output directories
RADAR_OUTPUT_DIR = "result/chart/radar/wind_rose"
TOWER_OUTPUT_DIR = "result/chart/tower/wind_rose"


def get_radar_speed_column(height):
    """Get the wind speed column name for radar data."""
    return f"Wind Speed{height}m"


def get_radar_direction_column(height):
    """Get the wind direction column name for radar data."""
    return f"Wind Direction{height}m"


def get_tower_speed_column(height):
    """Get the wind speed column name for tower data."""
    return f"Avg Wind Speed @ {height}m [m/s]"


def get_tower_direction_column(height):
    """Get the wind direction column name for tower data."""
    return f"Avg Wind Direction @ {height}m [deg]"


def compute_wind_energy(speed_ms):
    """
    Compute wind energy density from wind speed.

    Wind energy density (W/m^2) = 0.5 * rho * V^3
    Using standard air density rho = 1.225 kg/m^3

    Parameters
    ----------
    speed_ms : array-like
        Wind speed in m/s.

    Returns
    -------
    array-like
        Wind energy density in W/m^2.
    """
    rho = 1.225  # kg/m^3
    return 0.5 * rho * speed_ms ** 3


def create_rose_data(df, speed_col, direction_col, n_sectors=16):
    """
    Compute binned wind rose data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with wind speed and direction columns.
    speed_col : str
        Column name for wind speed.
    direction_col : str
        Column name for wind direction.
    n_sectors : int
        Number of directional sectors (default 16 for 22.5-degree bins).

    Returns
    -------
    dict
        Dictionary with 'direction_centers', 'frequency', 'mean_speed', 'mean_energy'.
    """
    sector_width = 360 / n_sectors

    # Clean data
    valid = df[[speed_col, direction_col]].dropna()
    valid = valid[valid[speed_col] >= 0]
    valid = valid[(valid[direction_col] >= 0) & (valid[direction_col] < 360)]

    if len(valid) == 0:
        return None

    # Bin directions into sectors
    sector_bins = np.arange(0, 360 + sector_width, sector_width)
    sector_labels = np.arange(n_sectors)

    valid['sector'] = np.digitize(valid[direction_col], sector_bins[1:-1])

    # Compute statistics per sector
    direction_centers = []
    frequency = []
    mean_speed = []
    mean_energy = []

    total_count = len(valid)

    for s in range(n_sectors):
        sector_data = valid[valid['sector'] == s]
        count = len(sector_data)

        if count > 0:
            direction_centers.append(s * sector_width + sector_width / 2)
            frequency.append(count / total_count * 100)  # Percentage
            mean_speed.append(sector_data[speed_col].mean())
            mean_energy.append(compute_wind_energy(sector_data[speed_col].values).mean())
        else:
            direction_centers.append(s * sector_width + sector_width / 2)
            frequency.append(0)
            mean_speed.append(0)
            mean_energy.append(0)

    return {
        'direction_centers': np.array(direction_centers),
        'frequency': np.array(frequency),
        'mean_speed': np.array(mean_speed),
        'mean_energy': np.array(mean_energy),
        'n_sectors': n_sectors,
        'total_samples': total_count
    }


def plot_wind_direction_rose(rose_data, title, output_path):
    """
    Plot wind direction rose (frequency by direction).

    Creates a polar bar chart showing the frequency of wind coming from
    each direction sector.
    """
    if rose_data is None:
        print("No data to plot.")
        return

    n_sectors = rose_data['n_sectors']
    sector_width = 2 * np.pi / n_sectors
    directions_rad = np.deg2rad(rose_data['direction_centers'])
    frequency = rose_data['frequency']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Set zero at North, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Plot bars
    bars = ax.bar(directions_rad, frequency, width=sector_width, bottom=0,
                  color=plt.cm.Blues(np.linspace(0.3, 0.9, n_sectors)),
                  edgecolor='white', linewidth=0.5)

    # Customize
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(frequency) * 1.2 if max(frequency) > 0 else 10)

    # Set direction labels
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10)

    # Add radial grid
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Frequency (%)', fontsize=10, labelpad=30)

    # Add sample count annotation
    info_text = f"n = {rose_data['total_samples']:,}"
    ax.text(0.5, 0.02, info_text, transform=ax.transAxes,
           ha='center', va='bottom', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_wind_energy_rose(rose_data, title, output_path):
    """
    Plot wind energy rose (mean energy by direction).

    Creates a polar bar chart showing the mean wind energy density
    from each direction sector.
    """
    if rose_data is None:
        print("No data to plot.")
        return

    n_sectors = rose_data['n_sectors']
    sector_width = 2 * np.pi / n_sectors
    directions_rad = np.deg2rad(rose_data['direction_centers'])
    mean_energy = rose_data['mean_energy'] / 1000  # Convert to kW/m^2

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Plot bars
    bars = ax.bar(directions_rad, mean_energy, width=sector_width, bottom=0,
                  color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, n_sectors)),
                  edgecolor='white', linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(mean_energy) * 1.2 if max(mean_energy) > 0 else 1)

    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Mean Energy Density (kW/m²)', fontsize=10, labelpad=30)

    info_text = f"n = {rose_data['total_samples']:,}"
    ax.text(0.5, 0.02, info_text, transform=ax.transAxes,
           ha='center', va='bottom', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_combined_rose(rose_data_list, titles, height_labels, plot_type, output_path):
    """
    Plot multiple heights on a single rose chart.

    Parameters
    ----------
    rose_data_list : list
        List of rose data dictionaries for different heights.
    titles : list
        List of height labels for legend.
    height_labels : list
        Display labels for each height.
    plot_type : str
        'frequency' or 'energy'.
    output_path : str
        Path to save the output figure.
    """
    if not rose_data_list:
        print("No data to plot.")
        return

    n_sectors = rose_data_list[0]['n_sectors']
    sector_width = 2 * np.pi / n_sectors
    directions_rad = np.deg2rad(rose_data_list[0]['direction_centers'])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    if plot_type == 'frequency':
        colors = plt.cm.cool(np.linspace(0.1, 0.9, len(rose_data_list)))
    else:
        colors = plt.cm.hot(np.linspace(0.1, 0.8, len(rose_data_list)))

    max_val = 0
    for rose_data, color in zip(rose_data_list, colors):
        if plot_type == 'frequency':
            values = rose_data['frequency']
            ylabel = 'Frequency (%)'
        else:
            values = rose_data['mean_energy'] / 1000
            ylabel = 'Mean Energy Density (kW/m²)'

        max_val = max(max_val, max(values) * 1.2 if max(values) > 0 else 1)

        ax.bar(directions_rad, values, width=sector_width, bottom=0,
              color=color, alpha=0.5, edgecolor=color, linewidth=1.5,
              label=rose_data.get('height_label', ''))

    ax.set_ylim(0, max_val)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=30)
    ax.legend(title='Height', loc='upper right', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_path}")


def process_and_plot(data_source, df, time_col, heights, speed_fn, direction_fn,
                     output_dir, source_name):
    """
    Process data and generate all rose plots for a data source.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select representative heights for combined plots
    if len(heights) <= 6:
        selected_heights = heights
    else:
        # Select evenly spaced heights
        step = len(heights) // 5
        selected_heights = heights[::step][:6]

    all_frequency_data = []
    all_energy_data = []

    for h in selected_heights:
        speed_col = speed_fn(h)
        direction_col = direction_fn(h)

        if speed_col not in df.columns or direction_col not in df.columns:
            continue

        rose_data = create_rose_data(df, speed_col, direction_col)

        if rose_data is None:
            continue

        rose_data['height_label'] = f'{h}m'

        # Individual plots
        plot_wind_direction_rose(
            rose_data,
            f"{source_name} Wind Direction Rose ({h}m)",
            os.path.join(output_dir, f"direction_rose_{h}m.png")
        )

        plot_wind_energy_rose(
            rose_data,
            f"{source_name} Wind Energy Rose ({h}m)",
            os.path.join(output_dir, f"energy_rose_{h}m.png")
        )

        all_frequency_data.append(rose_data)
        all_energy_data.append(rose_data)

    # Combined plots
    if all_frequency_data:
        plot_combined_rose(
            all_frequency_data,
            titles=[f"{h}m" for h in selected_heights],
            height_labels=[f"{h}m" for h in selected_heights],
            plot_type='frequency',
            output_path=os.path.join(output_dir, "direction_rose_combined.png")
        )

        plot_combined_rose(
            all_energy_data,
            titles=[f"{h}m" for h in selected_heights],
            height_labels=[f"{h}m" for h in selected_heights],
            plot_type='energy',
            output_path=os.path.join(output_dir, "energy_rose_combined.png")
        )


def main():
    """Generate wind rose and energy rose plots for radar and tower data."""
    os.makedirs(RADAR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOWER_OUTPUT_DIR, exist_ok=True)

    # Import radar data
    print("Loading radar data...")
    radar_df = import_data("data/radar.csv")
    radar_time_col = "Time" if "Time" in radar_df.columns else "timestamp"

    # Import tower data
    print("Loading tower data...")
    tower_df = import_data("data/tower.csv")
    tower_time_col = "Time" if "Time" in tower_df.columns else "timestamp"

    # Generate radar rose plots
    print("\nGenerating radar wind rose plots...")
    process_and_plot(
        data_source='radar',
        df=radar_df,
        time_col=radar_time_col,
        heights=RADAR_HEIGHTS,
        speed_fn=get_radar_speed_column,
        direction_fn=get_radar_direction_column,
        output_dir=RADAR_OUTPUT_DIR,
        source_name="Radar"
    )

    # Generate tower rose plots
    print("\nGenerating tower wind rose plots...")
    process_and_plot(
        data_source='tower',
        df=tower_df,
        time_col=tower_time_col,
        heights=TOWER_HEIGHTS,
        speed_fn=get_tower_speed_column,
        direction_fn=get_tower_direction_column,
        output_dir=TOWER_OUTPUT_DIR,
        source_name="Tower"
    )

    print("\nDone! All wind rose plots generated.")


if __name__ == "__main__":
    main()
