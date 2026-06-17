import numpy as np
import pandas as pd

HEIGHTS = list(range(40, 201, 5))

def init_qc_flags(df):
    """
    Initialize QC flag columns for all configured heights.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing measurement columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with `QC_{h}m` columns added and set to 0.
    """

    df = df.copy()

    for h in HEIGHTS:

        df[f"QC_{h}m"] = 0

    return df

def angle_diff(a, b):
    """
    Compute the minimal angular difference between two angles (in degrees).

    Handles wrap-around at 360 degrees and operates elementwise for arrays/Series.

    Parameters
    ----------
    a, b : array-like or scalar
        Angles in degrees.

    Returns
    -------
    array-like or scalar
        Minimal absolute difference between angles in degrees.
    """

    d = np.abs(a - b)

    return np.minimum(d, 360 - d)

def range_test(df):
    """
    Flag measurements outside physically plausible ranges.

    For each configured height this test marks QC as 1 when values are out of
    acceptable ranges: wind speed, wind direction, vertical wind speed, and standard deviation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with measurement columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with QC flags set to 1 where tests fail.
    """

    df = df.copy()

    for h in HEIGHTS:

        ws = f"Wind Speed{h}m"
        wd = f"Wind Direction{h}m"
        vw = f"Vertical Wind Speed{h}m"
        std = f"Wind Speed Std{h}m"

        qc = f"QC_{h}m"

        mask = pd.Series(False, index=df.index)

        if ws in df:
            mask |= (df[ws] < 0) | (df[ws] > 75)

        if wd in df:
            mask |= (df[wd] < 0) | (df[wd] >= 360)

        if vw in df:
            mask |= (df[vw] < -10) | (df[vw] > 10)

        if std in df:
            mask |= (df[std] < 0) | (df[std] > 20)

        df.loc[mask, qc] = 1

    return df

def turbulence_correlation_test(df):
    """
    Test turbulence intensity correlation and flag suspicious values.

    Computes turbulence intensity (TI = std / speed) and flags records where
    TI is negative or unrealistically large (>1) with QC code 2, preserving existing flags.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with wind speed and std columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 2 set where TI is invalid.
    """

    df = df.copy()

    for h in HEIGHTS:

        ws = f"Wind Speed{h}m"
        std = f"Wind Speed Std{h}m"
        qc = f"QC_{h}m"

        if ws not in df or std not in df:
            continue

        speed = df[ws]

        ti = df[std] / speed.replace(0, np.nan)

        mask = (
            (ti < 0)
            | (ti > 1)
        )

        df.loc[
            mask & (df[qc] == 0),
            qc
        ] = 2

    return df

def vertical_speed_correlation_test(df):
    """
    Flag large vertical differences in wind speed between adjacent heights.

    If the absolute difference between adjacent height wind speeds exceeds 15 m/s,
    set QC code 2 for both heights (if not already flagged).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with wind speed columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 2 set for inconsistent vertical speed pairs.
    """

    df = df.copy()

    for h in range(40, 200, 5):

        ws1 = f"Wind Speed{h}m"
        ws2 = f"Wind Speed{h+5}m"

        if ws1 not in df or ws2 not in df:
            continue

        diff = np.abs(
            df[ws2] - df[ws1]
        )

        mask = diff > 15

        df.loc[
            mask & (df[f"QC_{h}m"] == 0),
            f"QC_{h}m"
        ] = 2

        df.loc[
            mask & (df[f"QC_{h+5}m"] == 0),
            f"QC_{h+5}m"
        ] = 2

    return df

def vertical_direction_correlation_test(df):
    """
    Flag large direction differences between adjacent heights.

    Uses `angle_diff` to handle circular wrap-around and marks QC code 2 when
    the direction difference exceeds 120 degrees.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with wind direction columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 2 set for large direction discrepancies.
    """

    df = df.copy()

    for h in range(40, 200, 5):

        wd1 = f"Wind Direction{h}m"
        wd2 = f"Wind Direction{h+5}m"

        if wd1 not in df or wd2 not in df:
            continue

        diff = angle_diff(
            df[wd1],
            df[wd2]
        )

        mask = diff > 120

        df.loc[
            mask & (df[f"QC_{h}m"] == 0),
            f"QC_{h}m"
        ] = 2

        df.loc[
            mask & (df[f"QC_{h+5}m"] == 0),
            f"QC_{h+5}m"
        ] = 2

    return df

def flatline_test(df, window=6):
    """
    Detect flatline (constant) wind speed series over a rolling window.

    Marks QC code 3 for records where the wind speed is constant (rounded to 3 decimals)
    across the specified rolling window.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with wind speed columns.

    window : int, optional
        Rolling window size in observations. Default is 6.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 3 set where flatlines are detected.
    """

    df = df.copy()

    for h in HEIGHTS:

        ws = f"Wind Speed{h}m"

        if ws not in df:
            continue

        flat = (
            df[ws]
            .rolling(window)
            .apply(
                lambda x:
                len(set(np.round(x, 3))) == 1,
                raw=False
            )
            .fillna(0)
            .astype(bool)
        )

        df.loc[
            flat &
            (df[f"QC_{h}m"] == 0),
            f"QC_{h}m"
        ] = 3

    return df

def spike_test(df):
    """
    Detect sudden spikes in wind speed time series.

    Flags QC code 3 when the absolute change between consecutive samples exceeds 20 m/s.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with wind speed columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 3 set for spike detections.
    """

    df = df.copy()

    for h in HEIGHTS:

        ws = f"Wind Speed{h}m"

        if ws not in df:
            continue

        dv = np.abs(
            df[ws].diff()
        )

        mask = dv > 20

        df.loc[
            mask &
            (df[f"QC_{h}m"] == 0),
            f"QC_{h}m"
        ] = 3

    return df

def profile_consistency_test(df):
    """
    Check vertical profile consistency for each timestamp.

    For each profile (all heights) this test computes sign changes in the derivative
    of the wind speed profile and marks QC code 4 for profiles with excessive oscillations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with wind speed columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 4 set for inconsistent profiles.
    """

    df = df.copy()

    speed_cols = [
        f"Wind Speed{h}m"
        for h in HEIGHTS
        if f"Wind Speed{h}m" in df.columns
    ]

    for idx, row in df.iterrows():

        profile = row[speed_cols].values.astype(float)

        if np.sum(~np.isnan(profile)) < 10:
            continue

        d = np.diff(profile)

        d = d[~np.isnan(d)]

        if len(d) < 5:
            continue

        sign_changes = np.sum(
            np.diff(np.sign(d)) != 0
        )

        if sign_changes > 8:

            for h in HEIGHTS:

                qc = f"QC_{h}m"

                if df.at[idx, qc] == 0:
                    df.at[idx, qc] = 4

    return df

def missing_test(df):
    """
    Flag records with missing wind speed or direction at each height.

    Sets QC code 5 for heights where either wind speed or wind direction is missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with measurement columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with QC code 5 set for missing data.
    """

    df = df.copy()

    for h in HEIGHTS:

        qc = f"QC_{h}m"

        cols = [
            f"Wind Speed{h}m",
            f"Wind Direction{h}m"
        ]

        missing = pd.Series(False,index=df.index)

        for col in cols:

            if col in df:
                missing |= df[col].isna()

        df.loc[
            missing & (df[qc]==0),
            qc
        ] = 5

    return df

def apply_qc(df):
    """
    Apply QC flags by setting invalid measurements to NaN.

    For each height, if the QC flag is greater than 0, the corresponding
    measurement columns (speed, direction, vertical speed, std) are set to NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing QC flags and measurement columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with invalid measurements removed (set to NaN).
    """

    df = df.copy()

    for h in HEIGHTS:

        qc = f"QC_{h}m"

        cols = [
            f"Wind Speed{h}m",
            f"Wind Direction{h}m",
            f"Vertical Wind Speed{h}m",
            f"Wind Speed Std{h}m"
        ]

        for col in cols:

            if col not in df:
                continue

            df.loc[
                df[qc] > 0,
                col
            ] = np.nan

    return df

def track_step_availability(df, output_path="result/qc_radar_rate.csv"):
    """
    Track data availability after each QC step without modifying the input.

    Runs each QC test group sequentially (cumulatively) on a copy of the input
    DataFrame and records the percentage of records that remain valid
    (QC == 0) at each height after every stage. Also summarizes the QC code
    distribution and overall (mean across heights) availability.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw measurement DataFrame (no QC flag columns required).

    output_path : str, optional
        Path to write the availability summary CSV.

    Returns
    -------
    pandas.DataFrame
        The availability summary table (per-height rows plus a mean row).
    """

    stages = [
        ("After Missing Test(%)", [missing_test]),
        ("After Range Test(%)", [range_test]),
        ("After Correlation Test(%)", [
            turbulence_correlation_test,
            vertical_speed_correlation_test,
            vertical_direction_correlation_test,
        ]),
        ("After Trend Test(%)", [flatline_test, spike_test]),
    ]

    stage_labels = [
        "Raw Availability(%)",
        "After Missing Test(%)",
        "After Range Test(%)",
        "After Correlation Test(%)",
        "After Trend Test(%)",
        "Final Availability(%)",
    ]

    df_track = init_qc_flags(df)

    total_records = len(df_track)

    raw_avails = []
    stage_avails = {label: [] for label in stage_labels[1:-1]}

    for h in HEIGHTS:

        qc = f"QC_{h}m"
        ws = f"Wind Speed{h}m"

        if ws in df_track.columns:
            raw_avails.append(df_track[ws].notna().sum() / total_records * 100)
        else:
            raw_avails.append(0.0)

        for stage_label, funcs in stages:

            for func in funcs:
                df_track = func(df_track)

            stage_avails[stage_label].append(
                (df_track[qc] == 0).sum() / total_records * 100
            )

    final_avails = stage_avails["After Trend Test(%)"]

    qc_codes = [
        (1, "QC1 Range(%)"),
        (2, "QC2 Correlation(%)"),
        (3, "QC3 Trend(%)"),
        (4, "QC4 Profile(%)"),
    ]

    code_dists = {col: [] for _, col in qc_codes}

    for h in HEIGHTS:

        qc = f"QC_{h}m"

        for code, col in qc_codes:

            count = (df_track[qc] == code).sum()
            code_dists[col].append(count / total_records * 100)

    rows = []

    for i, h in enumerate(HEIGHTS):

        row = {
            "Height(m)": h,
            "Total Records": total_records,
            "Raw Availability(%)": round(raw_avails[i], 2),
            "After Missing Test(%)": round(stage_avails["After Missing Test(%)"][i], 2),
            "After Range Test(%)": round(stage_avails["After Range Test(%)"][i], 2),
            "After Correlation Test(%)": round(stage_avails["After Correlation Test(%)"][i], 2),
            "After Trend Test(%)": round(stage_avails["After Trend Test(%)"][i], 2),
            "Final Availability(%)": round(final_avails[i], 2),
        }

        for _, col in qc_codes:
            row[col] = round(code_dists[col][i], 2)

        rows.append(row)

    mean_row = {"Height(m)": "Mean", "Total Records": total_records}

    for key in stage_labels:

        if key == "Raw Availability(%)":
            vals = raw_avails
        elif key == "Final Availability(%)":
            vals = final_avails
        else:
            vals = stage_avails[key]

        mean_row[key] = round(sum(vals) / len(vals), 2)

    for _, col in qc_codes:
        mean_row[col] = round(sum(code_dists[col]) / len(code_dists[col]), 2)

    rows.append(mean_row)

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    return result

def radar_run_qc(df):
    """
    Run the full QC pipeline on the input DataFrame.

    This function performs initialization of QC flags, executes a sequence of
    tests (range, missing, correlation, trend, spike/profile tests), applies
    the QC by removing invalid values, and returns both the cleaned DataFrame
    and the QC flags DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw measurement DataFrame to be quality-controlled.

    Returns
    -------
    tuple
        `(qc_df, qc_df)` where `qc_df` has invalid measurements set to NaN,
        and `qc_df` contains the QC code columns.
    """

    df = init_qc_flags(df)

    print("Range Test...")
    df = range_test(df)
    
    print("Missing Test...")
    df = missing_test(df)

    print("Correlation Test...")
    df = turbulence_correlation_test(df)
    df = vertical_speed_correlation_test(df)
    df = vertical_direction_correlation_test(df)

    print("Trend Test...")
    df = flatline_test(df)
    df = spike_test(df)

    print("Profile Test...")
    # df = profile_consistency_test(df)

    print("Apply QC...")
    qc_df = apply_qc(df)

    return qc_df




