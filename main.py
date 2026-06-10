from src.data_file import *
from src.radar.qc_filter import radar_run_qc
from src.radar.qc_cat import radar_cat_qc
from src.tower.qc_filter import tower_run_qc
from src.tower.qc_cat import tower_cat_qc
import sys

def main():
    """
    Main program entry point.

    Imports radar and tower data, runs quality control (QC) and categorization
    for both datasets, then exports QC results to CSV files.

    Returns
    -------
    None
    """
    # Import radar data
    radar_data = import_data("data/radar.csv")
    if radar_data is not None:
        print("Data imported successfully. Here are the first few rows:")
        print(radar_data.head())
    else:
        print("Failed to import data.")
        sys.exit(1) 

    # Import tower data
    tower_data = import_data("data/tower.csv")
    if tower_data is not None:
        print("Tower data imported successfully. Here are the first few rows:")
        print(tower_data.head())
    else:
        print("Failed to import tower data.")
        sys.exit(1)
    
    # Run QC and categorize results
    qc_radar_data = radar_run_qc(radar_data)
    radar_cat_qc(radar_data, qc_radar_data)
    
    # TODO: Tower数据的处理方式不正确，表头不一样，当前按照雷达数据的方式处理，后续需要修改处理逻辑

    qc_tower_data = tower_run_qc(tower_data)
    tower_cat_qc(tower_data, qc_tower_data)
    
    # Export QC results
    export_data(qc_radar_data, "result/qc_radar.csv")
    export_data(qc_tower_data, "result/qc_tower.csv")


if __name__ == "__main__":
    main()
