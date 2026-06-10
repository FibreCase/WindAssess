from src.data_file import *
from src.qc_filter import run_qc
from src.qc_cat import cat_qc
import sys

def main():
    # Import radar data
    radar_data = import_data("data/radar.csv")
    if radar_data is not None:
        print("Data imported successfully. Here are the first few rows:")
        print(radar_data)
    else:
        print("Failed to import data.")
        sys.exit(1) 
    
    qc_radar_data, radar_data = run_qc(radar_data)
    print(qc_radar_data)
    cat_qc(radar_data, qc_radar_data)
    
    export_data(qc_radar_data, "result/qc_radar.csv")


if __name__ == "__main__":
    main()
