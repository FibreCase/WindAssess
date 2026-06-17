from src.data_file import *
from src.radar.qc_filter import radar_run_qc
from src.radar.qc_filter import track_step_availability as radar_track_step_availability
from src.radar.qc_cat import radar_cat_qc
from src.tower.qc_filter import tower_run_qc
from src.tower.qc_filter import track_step_availability as tower_track_step_availability
from src.tower.qc_cat import tower_cat_qc
from src.chart.weibull_plot import main as weibull_plot_main
from src.chart.daily_variation import main as daily_variation_main
from src.chart.density_variation import main as density_variation_main
from src.chart.shear_variation import main as shear_variation_main
from src.chart.wind_rose import main as wind_rose_main
import sys

def main():
    """
    Main program entry point.

    Imports radar and tower data, runs quality control (QC) and categorization
    for both datasets, exports QC results to CSV files, and generates charts.

    Returns
    -------
    None
    """

    # Import data
    print("\n------------------------------------------------")
    print("Starting data import...")
    print("------------------------------------------------\n")

    radar_data = import_data("data/radar.csv")
    if radar_data is not None:
        print("Data imported successfully. Here are the first few rows:")
        print(radar_data.head())
    else:
        print("Failed to import data.")
        sys.exit(1) 

    tower_data = import_data("data/tower.csv")
    if tower_data is not None:
        print("Tower data imported successfully. Here are the first few rows:")
        print(tower_data.head())
    else:
        print("Failed to import tower data.")
        sys.exit(1)
    
    # Run QC and categorize results
    print("\n------------------------------------------------")
    print("Starting QC and categorization for radar data...")
    print("------------------------------------------------\n")

    qc_radar_data = radar_run_qc(radar_data)
    radar_cat_qc(radar_data, qc_radar_data)

    print("\n------------------------------------------------")
    print("Tracking per-step availability for radar data...")
    print("------------------------------------------------\n")

    radar_rate = radar_track_step_availability(radar_data)
    print(radar_rate)

    print("\n------------------------------------------------")
    print("Starting QC and categorization for tower data...")
    print("------------------------------------------------\n")

    qc_tower_data = tower_run_qc(tower_data)
    tower_cat_qc(tower_data, qc_tower_data)

    print("\n------------------------------------------------")
    print("Tracking per-step availability for tower data...")
    print("------------------------------------------------\n")

    tower_rate = tower_track_step_availability(tower_data)
    print(tower_rate)
    
    # Export QC results
    print("\n------------------------------------------------")
    print("Exporting QC results to CSV files...")
    print("------------------------------------------------\n")
    
    export_data(qc_radar_data, "result/qc_radar.csv")
    export_data(qc_tower_data, "result/qc_tower.csv")

    # Generate charts
    print("\n------------------------------------------------")
    print("Generating Weibull distribution plots...")
    print("------------------------------------------------\n")
    
    weibull_plot_main()

    print("\n------------------------------------------------")
    print("Generating daily variation plots...")
    print("------------------------------------------------\n")
    
    daily_variation_main()

    print("\n------------------------------------------------")
    print("Generating air density variation plots...")
    print("------------------------------------------------\n")
    
    density_variation_main()

    print("\n------------------------------------------------")
    print("Generating wind shear variation plots...")
    print("------------------------------------------------\n")
    
    shear_variation_main()

    print("\n------------------------------------------------")
    print("Generating wind rose plots...")
    print("------------------------------------------------\n")
    
    wind_rose_main()

    print("\n================================================")
    print("All processing complete!")
    print("================================================")
    print("\nResults:")
    print("  - QC data: result/qc_radar.csv, result/qc_tower.csv")
    print("  - QC step availability: result/qc_radar_rate.csv, result/qc_tower_rate.csv")
    print("  - Radar charts:")
    print("    - Weibull: result/chart/radar/weibull/")
    print("    - Daily variation: result/chart/radar/daily_variation/")
    print("    - Density variation: result/chart/radar/density_variation/")
    print("    - Shear variation: result/chart/radar/shear_variation/")
    print("    - Wind rose: result/chart/radar/wind_rose/")
    print("  - Tower charts:")
    print("    - Weibull: result/chart/tower/weibull/")
    print("    - Daily variation: result/chart/tower/daily_variation/")
    print("    - Density variation: result/chart/tower/density_variation/")
    print("    - Shear variation: result/chart/tower/shear_variation/")
    print("    - Wind rose: result/chart/tower/wind_rose/")


if __name__ == "__main__":
    main()
