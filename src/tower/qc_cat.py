import numpy as np
import pandas as pd

# Tower data has specific heights: 2m, 5m, 10m, 20m, 50m, 80m
HEIGHTS = [2, 5, 10, 20, 50, 80]


def get_tower_column_name(height, var_type):
    """
    Get the correct column name for tower data based on height and variable type.

    Parameters
    ----------
    height : int
        Measurement height in meters (2, 5, 10, 20, 50, or 80).
    var_type : str
        Variable type: 'speed', 'direction', or 'std'.

    Returns
    -------
    str
        The column name string, or None if height is not valid.
    """
    if height not in HEIGHTS:
        return None

    if var_type == 'speed':
        return f"Avg Wind Speed @ {height}m [m/s]"
    elif var_type == 'direction':
        return f"Avg Wind Direction @ {height}m [deg]"
    elif var_type == 'std':
        return f"Avg Wind Speed (std dev) @ {height}m [m/s]"
    else:
        return None


def calculate_availability(df, heights=None):
    """
    Calculate the data availability after QC for each measurement height.
    """
    if heights is None:
        heights = HEIGHTS

    results = []
    total_records = len(df)

    for h in heights:
        qc_col = f"QC_{h}m"

        if qc_col not in df.columns:
            continue

        available = (df[qc_col] == 0).sum()
        availability = available / total_records * 100 if total_records > 0 else np.nan

        results.append({
            "Height(m)": h,
            "Total Records": total_records,
            "Available Records": available,
            "Availability(%)": round(availability, 2)
        })

    return pd.DataFrame(results)


def calculate_raw_availability(df, heights=None):
    """
    Calculate raw availability for each height based on non-missing wind speed records.
    """
    if heights is None:
        heights = HEIGHTS

    results = []
    total_records = len(df)

    for h in heights:
        ws_col = get_tower_column_name(h, 'speed')

        if ws_col is None or ws_col not in df.columns:
            continue

        available = df[ws_col].notna().sum()
        availability = available / total_records * 100 if total_records > 0 else np.nan

        results.append({
            "Height(m)": h,
            "Raw Availability(%)": round(availability, 2)
        })

    return pd.DataFrame(results)


def tower_cat_qc(df, qc_df):
    """Compute and print availability summaries for raw and QC-filtered tower data."""
    availability = calculate_availability(qc_df)
    raw_availability = calculate_raw_availability(df)

    print("\nData Availability After QC:")
    print(availability)

    print("\nRaw Data Availability:")
    print(raw_availability)