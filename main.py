from src.data_file import *
from src.qc_filter import run_qc
from src.qc_cat import cat_qc
import sys

def main():
    """
    Main program entry point.

    Imports radar data, runs quality control (QC) and categorization,
    then exports QC results to a CSV file.

    Returns
    -------
    None
    """
    # Import radar data
    radar_data = import_data("data/radar.csv")
    if radar_data is not None:
        print("Data imported successfully. Here are the first few rows:")
        print(radar_data)
    else:
        print("Failed to import data.")
        sys.exit(1) 
    
    # Run QC and categorize results
    qc_radar_data = run_qc(radar_data)
    print(qc_radar_data)
    cat_qc(radar_data, qc_radar_data)
    
    # Export QC results
    export_data(qc_radar_data, "result/qc_radar.csv")


if __name__ == "__main__":
    main()
