"""Tests that the saved model loads and predicts on a real sample row.

These need model/energy_model.joblib, which build_notebook.py creates. If it is
not there yet the tests skip with a clear message rather than failing.
"""

import os
import sys

import joblib
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import features as F

MODEL_PATH = os.path.join(ROOT, "model", "energy_model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "energydata_complete.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="run build_notebook.py first to create model/energy_model.joblib",
)


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


def test_bundle_has_expected_keys(bundle):
    for key in ("model", "feature_columns", "feature_medians", "metrics"):
        assert key in bundle
    assert bundle["feature_columns"] == F.feature_columns()


def test_predict_on_sample_row(bundle):
    cols = bundle["feature_columns"]
    df = F.build_features(F.load_raw(DATA_PATH))
    sample = df[cols].iloc[[100]]
    pred = bundle["model"].predict(sample)
    assert pred.shape == (1,)
    # Appliance energy is a positive quantity in a sane range.
    assert 0 < float(pred[0]) < 2000


def test_predict_from_medians(bundle):
    # A row built entirely from the stored medians must still predict cleanly.
    cols = bundle["feature_columns"]
    row = pd.DataFrame([[bundle["feature_medians"][c] for c in cols]], columns=cols)
    pred = float(bundle["model"].predict(row)[0])
    assert pred == pred  # not NaN
    assert pred > 0


def test_metrics_are_reasonable(bundle):
    # Sanity check that training actually produced a useful model.
    best_r2 = max(m["r2"] for m in bundle["metrics"].values())
    assert best_r2 > 0.3
