import numpy as np
import pandas as pd


def calculate_availability(df, heights=None):
    """
    Calculate the data availability after QC for each measurement height.
    """

    if heights is None:
        heights = list(range(40, 201, 5))

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
        heights = list(range(40, 201, 5))

    results = []
    total_records = len(df)

    for h in heights:

        ws_col = f"Wind Speed{h}m"

        if ws_col not in df.columns:
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