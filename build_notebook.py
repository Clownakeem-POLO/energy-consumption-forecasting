"""Build and execute analysis.ipynb programmatically.

I construct the notebook cell by cell with nbformat, then run every cell with
nbclient so the outputs and figures are real, not pasted in. Running this
script end to end produces:

  - analysis.ipynb with executed outputs embedded
  - figures/*.png saved along the way
  - model/energy_model.joblib, the trained model bundle

Run it from the project root with the project venv:

    python build_notebook.py
"""

from __future__ import annotations

import os

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "analysis.ipynb")


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


cells = []

cells.append(md(
    "# Appliance energy use forecasting\n"
    "\n"
    "I am predicting how much energy a home's appliances draw (in Wh) from "
    "indoor sensor readings, outdoor weather, and the time of day. The data is "
    "the UCI Appliances Energy Prediction set: about 19,700 readings taken "
    "every 10 minutes over roughly 4.5 months from a single house in Belgium.\n"
    "\n"
    "The thing I care about here is treating it as a time series. The raw "
    "columns are just sensor snapshots. The interesting signal is in *when* the "
    "reading happened and what energy use looked like a few minutes ago. So I "
    "engineer time features and lag features, then compare a plain linear model "
    "against gradient boosting using a time based split."
))

cells.append(code(
    "import os\n"
    "import sys\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import joblib\n"
    "\n"
    "from sklearn.linear_model import LinearRegression\n"
    "from sklearn.ensemble import HistGradientBoostingRegressor\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "\n"
    "# Let the notebook import the src package whether it runs from the root\n"
    "# or from inside a tool.\n"
    "sys.path.insert(0, os.path.abspath('.'))\n"
    "from src import features as F\n"
    "\n"
    "FIG_DIR = 'figures'\n"
    "MODEL_DIR = 'model'\n"
    "os.makedirs(FIG_DIR, exist_ok=True)\n"
    "os.makedirs(MODEL_DIR, exist_ok=True)\n"
    "\n"
    "sns.set_theme(style='whitegrid')\n"
    "plt.rcParams['figure.dpi'] = 110\n"
    "RANDOM_STATE = 42"
))

cells.append(md(
    "## Load the data\n"
    "\n"
    "I parse the timestamp into a real datetime and sort by time. Sorting "
    "matters because every lag feature later depends on rows being in order."
))

cells.append(code(
    "df = F.load_raw('data/energydata_complete.csv')\n"
    "print('shape:', df.shape)\n"
    "print('date range:', df['date'].min(), 'to', df['date'].max())\n"
    "df[['date', 'Appliances', 'lights', 'T1', 'RH_1', 'T_out']].head()"
))

cells.append(md(
    "## A first look at the target\n"
    "\n"
    "Appliance energy use is spiky. Most of the time the house sits at a low "
    "baseline, then jumps when someone cooks, does laundry, or runs something "
    "heavy. That long right tail is worth knowing about, it means average error "
    "will be dominated by how well the model handles the spikes."
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].hist(df['Appliances'], bins=60, color='#2c7fb8')\n"
    "axes[0].set_title('Appliance energy use (Wh)')\n"
    "axes[0].set_xlabel('Wh')\n"
    "axes[0].set_ylabel('count')\n"
    "\n"
    "# One week of the raw series so the daily rhythm is visible.\n"
    "week = df[df['date'] < df['date'].min() + pd.Timedelta(days=7)]\n"
    "axes[1].plot(week['date'], week['Appliances'], linewidth=0.8, color='#2c7fb8')\n"
    "axes[1].set_title('First week of readings')\n"
    "axes[1].set_xlabel('date')\n"
    "axes[1].set_ylabel('Wh')\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(FIG_DIR, 'target_overview.png'), bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(df['Appliances'].describe())"
))

cells.append(md(
    "## Feature engineering\n"
    "\n"
    "This is the core of the project. I build three kinds of features:\n"
    "\n"
    "- **Time features** from the timestamp: hour of day, day of week, a weekend "
    "flag, and month. These capture the daily and weekly pattern of when people "
    "are home and using things.\n"
    "- **Lag features** of the target: the reading 10 minutes ago, the reading "
    "one hour ago, and a rolling mean of the last hour. Energy use is sticky, so "
    "recent history is a strong predictor.\n"
    "- The **sensor and weather columns** that come with the data. I drop rv1 "
    "and rv2, which are two random noise columns the authors added as a control.\n"
    "\n"
    "All of this lives in `src/features.py` so the app and the tests use the "
    "exact same code."
))

cells.append(code(
    "feat = F.build_features(df)\n"
    "cols = F.feature_columns()\n"
    "print('rows after lag dropna:', len(feat))\n"
    "print('number of features:', len(cols))\n"
    "feat[['date', 'Appliances'] + F.TIME_COLUMNS + F.LAG_COLUMNS].head()"
))

cells.append(md(
    "## Why a time based split\n"
    "\n"
    "A random train/test split would let the model train on readings that come "
    "after the ones it is tested on. For a time series that is cheating: in the "
    "real world you never get to see the future. So I train on the first 80 "
    "percent of the timeline and test on the last 20 percent, kept in order. "
    "The scores below are lower than a random split would give, and that is the "
    "point, they are honest."
))

cells.append(code(
    "train, test = F.time_based_split(feat, train_frac=0.8)\n"
    "X_train, y_train = train[cols], train[F.TARGET]\n"
    "X_test, y_test = test[cols], test[F.TARGET]\n"
    "print('train rows:', len(train), '| test rows:', len(test))\n"
    "print('train ends:', train['date'].max())\n"
    "print('test starts:', test['date'].min())"
))

cells.append(md(
    "## Correlation with the target\n"
    "\n"
    "A quick scan of which features move with appliance use. The lag features "
    "stand out, which is the expected result and a good sign the time series "
    "framing is pulling its weight."
))

cells.append(code(
    "corr = feat[cols + [F.TARGET]].corr()[F.TARGET].drop(F.TARGET).sort_values()\n"
    "fig, ax = plt.subplots(figsize=(8, 9))\n"
    "colors = ['#d7301f' if v < 0 else '#2c7fb8' for v in corr.values]\n"
    "ax.barh(corr.index, corr.values, color=colors)\n"
    "ax.set_title('Correlation of each feature with appliance energy use')\n"
    "ax.set_xlabel('Pearson correlation')\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(FIG_DIR, 'feature_correlation.png'), bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md(
    "## Baseline: linear regression\n"
    "\n"
    "I start with a plain linear model. It is fast, it is a fair baseline, and "
    "if a heavier model cannot beat it then the heavier model is not worth the "
    "cost."
))

cells.append(code(
    "def evaluate(name, model, X_tr, y_tr, X_te, y_te):\n"
    "    model.fit(X_tr, y_tr)\n"
    "    pred = model.predict(X_te)\n"
    "    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))\n"
    "    mae = float(mean_absolute_error(y_te, pred))\n"
    "    r2 = float(r2_score(y_te, pred))\n"
    "    print(f'{name:>22}:  RMSE={rmse:7.2f}  MAE={mae:7.2f}  R2={r2:6.3f}')\n"
    "    return {'name': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'pred': pred}\n"
    "\n"
    "results = []\n"
    "lin = LinearRegression()\n"
    "results.append(evaluate('Linear regression', lin, X_train, y_train, X_test, y_test))"
))

cells.append(md(
    "## Gradient boosting\n"
    "\n"
    "Histogram gradient boosting. In theory it can catch the non linear "
    "interactions between time of day, weather, and recent use that the linear "
    "model cannot. I keep it shallow and use early stopping so it does not just "
    "memorize the training spikes. I tried deeper and faster settings too and "
    "they all did worse on the held out period, so shallow and slow it is."
))

cells.append(code(
    "hgb = HistGradientBoostingRegressor(\n"
    "    max_iter=500,\n"
    "    learning_rate=0.03,\n"
    "    max_depth=3,\n"
    "    l2_regularization=1.0,\n"
    "    early_stopping=True,\n"
    "    validation_fraction=0.15,\n"
    "    n_iter_no_change=20,\n"
    "    random_state=RANDOM_STATE,\n"
    ")\n"
    "results.append(evaluate('Gradient boosting', hgb, X_train, y_train, X_test, y_test))"
))

cells.append(md(
    "## Results side by side\n"
    "\n"
    "Here is the honest surprise of this project: the linear baseline wins. On a "
    "time based split the lag features do most of the work and their relationship "
    "to the target is close to linear, so the extra flexibility of gradient "
    "boosting mostly buys it a chance to overfit. I would rather report this than "
    "quietly swap in a fancier model that does worse. I pick whichever model has "
    "the lower RMSE as the one to ship."
))

cells.append(code(
    "summary = pd.DataFrame([{k: r[k] for k in ('name', 'rmse', 'mae', 'r2')} for r in results])\n"
    "summary = summary.set_index('name').round(3)\n"
    "best = min(results, key=lambda r: r['rmse'])\n"
    "best_model = lin if best['name'] == 'Linear regression' else hgb\n"
    "print('best model by RMSE:', best['name'])\n"
    "summary"
))

cells.append(md(
    "## Feature importance\n"
    "\n"
    "Permutation importance on the best model. I shuffle each feature one at a "
    "time and watch how much the score drops. It works on any model, not just "
    "trees, and it is honest about what the model actually leans on. The lag "
    "features dominate, which is exactly why the time series framing matters."
))

cells.append(code(
    "from sklearn.inspection import permutation_importance\n"
    "\n"
    "perm = permutation_importance(\n"
    "    best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1\n"
    ")\n"
    "imp = pd.Series(perm.importances_mean, index=cols).sort_values().tail(15)\n"
    "fig, ax = plt.subplots(figsize=(8, 7))\n"
    "ax.barh(imp.index, imp.values, color='#238b45')\n"
    "ax.set_title('Permutation importance (best model, top 15)')\n"
    "ax.set_xlabel('drop in score when feature is shuffled')\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md(
    "## Actual vs predicted over time\n"
    "\n"
    "The numbers only say so much. Here is the best model's prediction laid over "
    "the real series for a slice of the held out test period. It tracks the "
    "baseline and the timing of spikes well, and like most models on spiky data "
    "it tends to undershoot the tallest peaks."
))

cells.append(code(
    "best_pred = best['pred']\n"
    "plot_df = test[['date']].copy()\n"
    "plot_df['actual'] = y_test.values\n"
    "plot_df['predicted'] = best_pred\n"
    "window = plot_df.iloc[:600]  # about 4 days at 10 minute spacing\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(13, 4.5))\n"
    "ax.plot(window['date'], window['actual'], label='actual', linewidth=1.0, color='#333333')\n"
    "ax.plot(window['date'], window['predicted'], label='predicted', linewidth=1.0, color='#d7301f', alpha=0.8)\n"
    "ax.set_title('Actual vs predicted appliance energy use (held out test window)')\n"
    "ax.set_xlabel('date')\n"
    "ax.set_ylabel('Wh')\n"
    "ax.legend()\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(FIG_DIR, 'actual_vs_predicted.png'), bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md(
    "## Residuals\n"
    "\n"
    "Residuals are actual minus predicted. I want them centered near zero with "
    "no strong pattern against the prediction. The spread widens at higher "
    "predicted values, which is the undershoot on spikes showing up again."
))

cells.append(code(
    "resid = y_test.values - best_pred\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].hist(resid, bins=60, color='#756bb1')\n"
    "axes[0].axvline(0, color='k', linewidth=1)\n"
    "axes[0].set_title('Residual distribution')\n"
    "axes[0].set_xlabel('actual - predicted (Wh)')\n"
    "\n"
    "axes[1].scatter(best_pred, resid, s=5, alpha=0.3, color='#756bb1')\n"
    "axes[1].axhline(0, color='k', linewidth=1)\n"
    "axes[1].set_title('Residuals vs predicted')\n"
    "axes[1].set_xlabel('predicted (Wh)')\n"
    "axes[1].set_ylabel('residual (Wh)')\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(FIG_DIR, 'residuals.png'), bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md(
    "## Save the model\n"
    "\n"
    "I retrain the winning model (the linear one) on the full timeline and save "
    "it as a bundle together with the ordered feature list and the training "
    "median of each feature. The app uses the medians to prefill sensible "
    "defaults so a user only has to change the few inputs they care about."
))

cells.append(code(
    "from sklearn.base import clone\n"
    "final_model = clone(best_model)\n"
    "final_model.fit(feat[cols], feat[F.TARGET])\n"
    "print('shipping model:', best['name'])\n"
    "\n"
    "bundle = {\n"
    "    'model': final_model,\n"
    "    'model_name': best['name'],\n"
    "    'feature_columns': cols,\n"
    "    'feature_medians': feat[cols].median().to_dict(),\n"
    "    'metrics': {r['name']: {k: r[k] for k in ('rmse', 'mae', 'r2')} for r in results},\n"
    "}\n"
    "out = os.path.join(MODEL_DIR, 'energy_model.joblib')\n"
    "joblib.dump(bundle, out)\n"
    "print('saved', out)\n"
    "print('bundle keys:', list(bundle.keys()))"
))

cells.append(md(
    "## What I would do next\n"
    "\n"
    "- The spikes are the hard part. A quantile loss or a model aimed at the "
    "peaks would help if the goal were catching demand surges.\n"
    "- There is only one house here, so the model learns this household's "
    "habits. It would need retraining per home.\n"
    "- Longer lags and a rolling std could add a little, but I kept the feature "
    "set small on purpose so it stays easy to read and fast to serve."
))


def main():
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": HERE}},
    )
    print("executing notebook...")
    client.execute()
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    print("wrote", NB_PATH)


if __name__ == "__main__":
    main()
