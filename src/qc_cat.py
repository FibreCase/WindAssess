import pandas as pd
import numpy as np

def calculate_availability(df, heights=None):
    """
    Calculate the data availability after QC for each measurement height.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing QC_{h}m columns indicating QC flags.

    heights : list[int], optional
        List of heights to evaluate. If None, defaults to 40..200 m every 5 m.

    Returns
    -------
    pandas.DataFrame
        Summary table with columns: "Height(m)", "Total Records", "Available Records", "Availability(%)".
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

        availability = (
            available / total_records * 100
            if total_records > 0
            else np.nan
        )

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

    Raw availability is defined as the percentage of non-NaN wind speed values
    at a given height divided by the total number of records.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with raw wind speed columns named like "Wind Speed{h}m".

    heights : list[int], optional
        Heights to include; defaults to 40..200 m every 5 m.

    Returns
    -------
    pandas.DataFrame
        Summary table with "Height(m)" and "Raw Availability(%)".
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

        availability = (
            available / total_records * 100
            if total_records > 0
            else np.nan
        )

        results.append({
            "Height(m)": h,
            "Raw Availability(%)": round(availability, 2)
        })

    return pd.DataFrame(results)

def cat_qc(df, qc_df):
    """
    Compute and print availability summaries for raw and QC-filtered data.

    This function prints two tables: availability after QC (using `qc_df`)
    and raw data availability (using `df`).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw measurement DataFrame containing wind speed columns.

    qc_df : pandas.DataFrame
        DataFrame containing QC flag columns `QC_{h}m`.

    Returns
    -------
    None
    """

    availability = calculate_availability(qc_df)
    raw_availability = calculate_raw_availability(df)

    print("\nData Availability After QC:")
    print(availability)

    print("\nRaw Data Availability:")
    print(raw_availability)