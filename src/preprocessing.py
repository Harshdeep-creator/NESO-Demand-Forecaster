import os
from pathlib import Path
import pandas as pd


RAW_DATA_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
OUTPUT_FILE = "demand_daily_2019_2026.csv"


# ------------------------------------------------
# LOAD & COMBINE
# ------------------------------------------------
def load_and_combine_raw_data(raw_folder):
    files = sorted(Path(raw_folder).glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found")

    dfs = []
    for file in files:
        print(f"Loading {file.name}")
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)
    print("Combined shape:", df.shape)
    return df


# ------------------------------------------------
# CREATE DATETIME INDEX
# ------------------------------------------------
def create_datetime_index(df):

    # Select only required columns and force copy
    df = df.loc[:, ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]].copy()

    # Clean types
    df.loc[:, "SETTLEMENT_DATE"] = pd.to_datetime(
        df["SETTLEMENT_DATE"],
        format="%d-%b-%Y",
        errors="coerce"
    )

    df.loc[:, "ND"] = pd.to_numeric(df["ND"], errors="coerce")

    df = df.dropna(subset=["SETTLEMENT_DATE"])

    # Remove exact duplicate rows first
    df = df.drop_duplicates()

    # Aggregate duplicate date + period
    df = (
        df.groupby(
            ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
            as_index=False
        )
        .agg({"ND": "mean"})
    )

    # Create datetime
    df["datetime"] = (
        df["SETTLEMENT_DATE"]
        + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="min")
    )

    df = df.set_index("datetime").sort_index()

    # 🔴 HARD SAFETY CHECK
    if df.index.duplicated().any():
        print("Duplicate timestamps detected. Removing...")
        df = df[~df.index.duplicated(keep="first")]

    print("Rows after cleaning:", len(df))
    print("Duplicate timestamps remaining:", df.index.duplicated().sum())

    return df


# ------------------------------------------------
# KEEP REQUIRED COLUMN
# ------------------------------------------------
def keep_required_columns(df):
    df = df.loc[:, ["ND"]].copy()
    df.columns = ["demand"]
    return df


# ------------------------------------------------
# FIX MISSING TIMESTAMPS
# ------------------------------------------------
def check_and_fix_missing(df):

    # 🔴 ENSURE UNIQUE INDEX BEFORE REINDEX
    if df.index.duplicated().any():
        print("Fixing duplicate index before reindex...")
        df = df[~df.index.duplicated(keep="first")]

    # Confirm uniqueness
    assert not df.index.duplicated().any(), "Index still contains duplicates!"

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="30min"
    )

    missing = full_range.difference(df.index)
    print("Missing timestamps:", len(missing))

    df = df.reindex(full_range)

    # Interpolate safely
    df["demand"] = df["demand"].interpolate(method="time")
    df["demand"] = df["demand"].bfill().ffill()

    return df


# ------------------------------------------------
# RESAMPLE TO DAILY
# ------------------------------------------------
def resample_to_daily(df):
    df_daily = df.resample("D").mean()
    print("Daily dataset shape:", df_daily.shape)
    return df_daily


# ------------------------------------------------
# SAVE
# ------------------------------------------------
def save_processed(df):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_PATH, OUTPUT_FILE)
    df.to_csv(path)
    print("Saved to:", path)


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():

    print("Step 1: Loading raw data...")
    df = load_and_combine_raw_data(RAW_DATA_PATH)

    print("Step 2: Creating datetime index...")
    df = create_datetime_index(df)

    print("Step 3: Keeping required columns...")
    df = keep_required_columns(df)

    print("Step 4: Fixing missing timestamps...")
    df = check_and_fix_missing(df)

    print("Step 5: Resampling to daily...")
    df_daily = resample_to_daily(df)

    print("Step 6: Saving...")
    save_processed(df_daily)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()