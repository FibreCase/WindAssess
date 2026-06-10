import numpy as np
import pandas as pd

HEIGHTS = list(range(40, 201, 5))

def init_qc_flags(df):

    df = df.copy()

    for h in HEIGHTS:

        df[f"QC_{h}m"] = 0

    return df

def angle_diff(a, b):

    d = np.abs(a - b)

    return np.minimum(d, 360 - d)

def range_test(df):

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

def run_qc(df):

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
    cleaned_df = apply_qc(df)

    return cleaned_df, df




