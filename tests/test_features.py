"""Tests for the feature engineering functions.

These run without the trained model, they only exercise src/features.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import features as F


@pytest.fixture
def toy_df():
    """A small hand built frame with a known timestamp pattern."""
    dates = pd.date_range("2016-01-11 17:00:00", periods=20, freq="10min")
    return pd.DataFrame({
        "date": dates,
        "Appliances": np.arange(20) * 10 + 50,
        "lights": 0,
        "T_out": 5.0,
        "RH_1": 40.0,
    })


def test_add_time_features_values(toy_df):
    out = F.add_time_features(toy_df)
    # First row is 2016-01-11 17:00, a Monday.
    assert out.loc[0, "hour"] == 17
    assert out.loc[0, "dayofweek"] == 0
    assert out.loc[0, "is_weekend"] == 0
    assert out.loc[0, "month"] == 1


def test_is_weekend_flag():
    # 2016-01-16 is a Saturday.
    df = pd.DataFrame({"date": pd.to_datetime(["2016-01-16 09:00:00"]), "Appliances": [100]})
    out = F.add_time_features(df)
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "dayofweek"] == 5


def test_lag_features_shift(toy_df):
    out = F.add_lag_features(toy_df)
    # lag_1 of row 1 should equal the target of row 0.
    assert out.loc[1, "Appliances_lag_1"] == toy_df.loc[0, "Appliances"]
    # lag_6 of row 6 should equal the target of row 0.
    assert out.loc[6, "Appliances_lag_6"] == toy_df.loc[0, "Appliances"]
    # The first row has no history, so its lag_1 is NaN.
    assert np.isnan(out.loc[0, "Appliances_lag_1"])


def test_build_features_drops_incomplete_rows(toy_df):
    out = F.build_features(toy_df)
    # No NaNs left in the lag columns after the pipeline.
    assert out[F.LAG_COLUMNS].isna().sum().sum() == 0
    # We lose the first 6 rows to the rolling/lag window.
    assert len(out) == len(toy_df) - 6


def test_feature_columns_no_leakage():
    cols = F.feature_columns()
    # The raw target must never appear as an input feature.
    assert F.TARGET not in cols
    # The two random noise columns must be excluded.
    assert "rv1" not in cols
    assert "rv2" not in cols
    # No duplicates.
    assert len(cols) == len(set(cols))


def test_time_based_split_is_ordered(toy_df):
    feat = F.build_features(toy_df)
    train, test = F.time_based_split(feat, train_frac=0.7)
    # Split preserves order: every train timestamp precedes every test one.
    assert train["date"].max() <= test["date"].min()
    assert len(train) + len(test) == len(feat)
