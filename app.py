"""Streamlit demo for the appliance energy forecasting model.

A user sets a few conditions (time of day, recent energy use, indoor and
outdoor readings) and gets a predicted appliance energy draw in Wh. Every
feature the model expects is prefilled with its training median, so the user
only changes what they care about.

Run it with:

    streamlit run app.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st

from src import features as F

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "energy_model.joblib")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def predict_one(bundle: dict, overrides: dict) -> float:
    """Build a single feature row from medians plus the user's overrides."""
    cols = bundle["feature_columns"]
    row = dict(bundle["feature_medians"])
    row.update(overrides)
    X = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    return float(bundle["model"].predict(X)[0])


def main():
    st.set_page_config(page_title="Appliance energy forecast", page_icon="plug")
    st.title("Appliance energy use forecast")
    st.write(
        "This predicts how much energy a home's appliances draw, in Wh, for a "
        "10 minute window. It is trained on the UCI Appliances Energy Prediction "
        "data, one house in Belgium sampled every 10 minutes. Set the conditions "
        "below and read the prediction. Everything you do not touch sits at the "
        "training median."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "I could not find the trained model. Run `python build_notebook.py` "
            "first to train and save it."
        )
        return

    bundle = load_bundle()
    medians = bundle["feature_medians"]

    st.caption(f"Model in use: {bundle.get('model_name', 'regression model')}")

    st.subheader("When")
    c1, c2 = st.columns(2)
    hour = c1.slider("Hour of day", 0, 23, 18)
    dow = c2.selectbox(
        "Day of week",
        options=list(range(7)),
        index=2,
        format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
    )
    month = c1.slider("Month", 1, 12, int(medians["month"]))
    is_weekend = 1 if dow >= 5 else 0

    st.subheader("Recent energy use")
    st.caption(
        "These are the strongest predictors. Energy use tends to stay close to "
        "what it just was."
    )
    c3, c4 = st.columns(2)
    lag_1 = c3.number_input(
        "Use 10 minutes ago (Wh)", min_value=0.0, max_value=1200.0,
        value=float(medians["Appliances_lag_1"]), step=10.0,
    )
    lag_6 = c4.number_input(
        "Use one hour ago (Wh)", min_value=0.0, max_value=1200.0,
        value=float(medians["Appliances_lag_6"]), step=10.0,
    )
    roll_6 = c3.number_input(
        "Average of last hour (Wh)", min_value=0.0, max_value=1200.0,
        value=float(medians["Appliances_roll_mean_6"]), step=10.0,
    )

    st.subheader("Conditions")
    c5, c6 = st.columns(2)
    t_out = c5.number_input(
        "Outdoor temperature (C)", min_value=-10.0, max_value=40.0,
        value=float(medians["T_out"]), step=0.5,
    )
    t1 = c6.number_input(
        "Kitchen temperature (C)", min_value=10.0, max_value=35.0,
        value=float(medians["T1"]), step=0.5,
    )
    rh_1 = c5.number_input(
        "Kitchen humidity (%)", min_value=10.0, max_value=70.0,
        value=float(medians["RH_1"]), step=1.0,
    )
    lights = c6.number_input(
        "Lights energy (Wh)", min_value=0.0, max_value=70.0,
        value=float(medians["lights"]), step=10.0,
    )

    overrides = {
        "hour": hour,
        "dayofweek": dow,
        "is_weekend": is_weekend,
        "month": month,
        "Appliances_lag_1": lag_1,
        "Appliances_lag_6": lag_6,
        "Appliances_roll_mean_6": roll_6,
        "T_out": t_out,
        "T1": t1,
        "RH_1": rh_1,
        "lights": lights,
    }

    pred = predict_one(bundle, overrides)
    st.subheader("Prediction")
    st.metric("Predicted appliance energy use", f"{pred:,.0f} Wh")
    st.caption(
        "A reminder from the writeup: on held out data this model lands within "
        "about 28 Wh on average, but it undershoots the biggest spikes. Treat it "
        "as a guide, not a guarantee."
    )


if __name__ == "__main__":
    main()
