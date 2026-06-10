import pandas as pd
import numpy as np

def calculate_availability(df, heights=None):
    """
    统计各高度数据可用率

    Parameters
    ----------
    df : pandas.DataFrame
        包含 QC_xxm 列的数据表

    heights : list[int], optional
        高度列表，默认40~200m，每5m一层

    Returns
    -------
    pandas.DataFrame
        Height(m)
        Total Records
        Available Records
        Availability(%)
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
    原始数据可用率
    = 非NaN风速记录数 / 总记录数
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
    availability = calculate_availability(qc_df)
    raw_availability = calculate_raw_availability(df)

    print("\nData Availability After QC:")
    print(availability)

    print("\nRaw Data Availability:")
    print(raw_availability)