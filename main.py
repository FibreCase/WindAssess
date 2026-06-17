from src.data_file import *
from src.radar.qc_filter import radar_run_qc
from src.radar.qc_cat import radar_cat_qc
from src.tower.qc_filter import tower_run_qc
from src.tower.qc_cat import tower_cat_qc
from src.chart.weibull_plot import main as weibull_plot_main
from src.chart.daily_variation import main as daily_variation_main
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
    print("Starting QC and categorization for tower data...")
    print("------------------------------------------------\n")

    qc_tower_data = tower_run_qc(tower_data)
    tower_cat_qc(tower_data, qc_tower_data)
    
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

    print("\n================================================")
    print("All processing complete!")
    print("================================================")
    print("\nResults:")
    print("  - QC data: result/qc_radar.csv, result/qc_tower.csv")
    print("  - Weibull charts: result/chart/weibull/")
    print("  - Daily variation charts: result/chart/daily_variation/")


if __name__ == "__main__":
    main()
