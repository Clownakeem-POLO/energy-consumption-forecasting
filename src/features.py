"""Feature engineering for the appliance energy forecasting project.

I keep the reusable logic here so it can be imported by the notebook, the
Streamlit app, and the tests. Nothing in here touches the model or the
plots, it is all pure data transforms on a pandas DataFrame.
"""

from __future__ import annotations

import pandas as pd

# The sensor and weather columns I feed the model. I drop rv1 and rv2 on
# purpose. They are two random noise columns the dataset authors added as a
# sanity check, so they carry no real signal.
SENSOR_COLUMNS = [
    "lights",
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
]

# Time features I build from the timestamp.
TIME_COLUMNS = ["hour", "dayofweek", "is_weekend", "month"]

# Lag and rolling features on the target itself.
LAG_COLUMNS = ["Appliances_lag_1", "Appliances_lag_6", "Appliances_roll_mean_6"]

TARGET = "Appliances"


def load_raw(csv_path: str) -> pd.DataFrame:
    """Read the raw CSV and parse the date column into a real datetime."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pull hour, day of week, weekend flag, and month out of the timestamp.

    These are the features that let a plain regression model learn the daily
    and weekly rhythm of a household, which is the whole point of treating
    this as a time series instead of a bag of sensor readings.
    """
    df = df.copy()
    ts = pd.to_datetime(df["date"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.day_of_week
    df["is_weekend"] = (ts.dt.day_of_week >= 5).astype(int)
    df["month"] = ts.dt.month
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling values of the target.

    lag_1 is the reading 10 minutes ago, lag_6 is one hour ago, and the
    rolling mean smooths the last hour. Energy use is sticky, so what just
    happened is a strong hint about what happens next. I build these after
    sorting by time so no future information leaks backward.
    """
    df = df.copy()
    df["Appliances_lag_1"] = df[TARGET].shift(1)
    df["Appliances_lag_6"] = df[TARGET].shift(6)
    df["Appliances_roll_mean_6"] = df[TARGET].shift(1).rolling(window=6).mean()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline and drop rows left incomplete by lags.

    The first few rows have no history to look back on, so their lag columns
    are NaN. I drop those rather than fill them, because inventing a value
    there would just be noise.
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = df.dropna(subset=LAG_COLUMNS).reset_index(drop=True)
    return df


def feature_columns() -> list[str]:
    """The exact ordered list of columns the model is trained on."""
    return SENSOR_COLUMNS + TIME_COLUMNS + LAG_COLUMNS


def time_based_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split by time, not at random.

    For a time series a random split lets the model peek at the future, which
    flatters the test score and would never happen in production. I cut the
    first 80 percent of the timeline for training and keep the last 20 percent,
    held out in order, for the test.
    """
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test
