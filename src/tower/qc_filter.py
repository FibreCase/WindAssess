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
    acceptable ranges: wind speed, wind direction, and standard deviation.
    Note: Tower data does not have vertical wind speed measurements.

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
        ws = get_tower_column_name(h, 'speed')
        wd = get_tower_column_name(h, 'direction')
        std = get_tower_column_name(h, 'std')

        qc = f"QC_{h}m"

        mask = pd.Series(False, index=df.index)

        if ws in df.columns:
            mask |= (df[ws] < 0) | (df[ws] > 75)

        if wd in df.columns:
            mask |= (df[wd] < 0) | (df[wd] >= 360)

        # Tower data does not have vertical wind speed

        if std in df.columns:
            mask |= (df[std] < 0) | (df[std] > 20)

        df.loc[mask, qc] = 1

    return df


def turbulence_correlation_test(df):
    """
    Test turbulence intensity correlation and flag suspicious values.

    Computes turbulence intensity (TI = std / speed) and flags records where
    TI is negative or unrealistically large (>1) with QC code 2, preserving existing flags.
    """
    df = df.copy()

    for h in HEIGHTS:
        ws = get_tower_column_name(h, 'speed')
        std = get_tower_column_name(h, 'std')
        qc = f"QC_{h}m"

        if ws not in df.columns or std not in df.columns:
            continue

        speed = df[ws]
        ti = df[std] / speed.replace(0, np.nan)

        mask = (ti < 0) | (ti > 1)

        df.loc[mask & (df[qc] == 0), qc] = 2

    return df


def vertical_speed_correlation_test(df):
    """
    Flag large vertical differences in wind speed between available heights.

    For tower data, compares between consecutive available heights rather than
    fixed 5m intervals.
    """
    df = df.copy()

    # Compare between consecutive available heights
    for i in range(len(HEIGHTS) - 1):
        h1 = HEIGHTS[i]
        h2 = HEIGHTS[i + 1]

        ws1 = get_tower_column_name(h1, 'speed')
        ws2 = get_tower_column_name(h2, 'speed')

        if ws1 not in df.columns or ws2 not in df.columns:
            continue

        diff = np.abs(df[ws2] - df[ws1])
        mask = diff > 15

        df.loc[mask & (df[f"QC_{h1}m"] == 0), f"QC_{h1}m"] = 2
        df.loc[mask & (df[f"QC_{h2}m"] == 0), f"QC_{h2}m"] = 2

    return df


def vertical_direction_correlation_test(df):
    """Flag large direction differences between available heights."""
    df = df.copy()

    # Compare between consecutive available heights
    for i in range(len(HEIGHTS) - 1):
        h1 = HEIGHTS[i]
        h2 = HEIGHTS[i + 1]

        wd1 = get_tower_column_name(h1, 'direction')
        wd2 = get_tower_column_name(h2, 'direction')

        if wd1 not in df.columns or wd2 not in df.columns:
            continue

        diff = angle_diff(df[wd1], df[wd2])
        mask = diff > 120

        df.loc[mask & (df[f"QC_{h1}m"] == 0), f"QC_{h1}m"] = 2
        df.loc[mask & (df[f"QC_{h2}m"] == 0), f"QC_{h2}m"] = 2

    return df


def flatline_test(df, window=6):
    """Detect flatline (constant) wind speed series over a rolling window."""
    df = df.copy()

    for h in HEIGHTS:
        ws = get_tower_column_name(h, 'speed')

        if ws not in df.columns:
            continue

        flat = (
            df[ws]
            .rolling(window)
            .apply(lambda x: len(set(np.round(x, 3))) == 1, raw=False)
            .fillna(0)
            .astype(bool)
        )

        df.loc[flat & (df[f"QC_{h}m"] == 0), f"QC_{h}m"] = 3

    return df


def spike_test(df):
    """Detect sudden spikes in wind speed time series."""
    df = df.copy()

    for h in HEIGHTS:
        ws = get_tower_column_name(h, 'speed')

        if ws not in df.columns:
            continue

        dv = np.abs(df[ws].diff())
        mask = dv > 20

        df.loc[mask & (df[f"QC_{h}m"] == 0), f"QC_{h}m"] = 3

    return df


def profile_consistency_test(df):
    """Check vertical profile consistency for each timestamp."""
    df = df.copy()

    speed_cols = [
        get_tower_column_name(h, 'speed')
        for h in HEIGHTS
        if get_tower_column_name(h, 'speed') in df.columns
    ]

    for idx, row in df.iterrows():
        profile = row[speed_cols].values.astype(float)

        if np.sum(~np.isnan(profile)) < 4:
            continue

        d = np.diff(profile)
        d = d[~np.isnan(d)]

        if len(d) < 3:
            continue

        sign_changes = np.sum(np.diff(np.sign(d)) != 0)

        # Lower threshold due to fewer height levels
        if sign_changes > 4:
            for h in HEIGHTS:
                qc = f"QC_{h}m"
                if df.at[idx, qc] == 0:
                    df.at[idx, qc] = 4

    return df


def missing_test(df):
    """Flag records with missing wind speed or direction at each height."""
    df = df.copy()

    for h in HEIGHTS:
        qc = f"QC_{h}m"
        ws = get_tower_column_name(h, 'speed')
        wd = get_tower_column_name(h, 'direction')

        cols = [ws, wd]
        missing = pd.Series(False, index=df.index)

        for col in cols:
            if col in df.columns:
                missing |= df[col].isna()

        df.loc[missing & (df[qc] == 0), qc] = 5

    return df


def apply_qc(df):
    """Apply QC flags by setting invalid measurements to NaN."""
    df = df.copy()

    for h in HEIGHTS:
        qc = f"QC_{h}m"

        cols = [
            get_tower_column_name(h, 'speed'),
            get_tower_column_name(h, 'direction'),
            get_tower_column_name(h, 'std'),
        ]

        for col in cols:
            if col is None or col not in df.columns:
                continue

            df.loc[df[qc] > 0, col] = np.nan

    return df


def track_step_availability(df, output_path="result/qc_tower_rate.csv"):
    """
    Track data availability after each QC step for tower data.

    See `src/radar/qc_filter.py:track_step_availability` for a detailed
    description. This tower variant uses the tower height list and the
    tower column naming convention.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw tower measurement DataFrame.

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
        ws = get_tower_column_name(h, 'speed')

        if ws is not None and ws in df_track.columns:
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


def tower_run_qc(df):
    """Run the full QC pipeline on the tower DataFrame."""
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