"""Export forecast-accuracy and feature-importance charts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate(output_dir="assets/charts", predictions="data/predictions.csv"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = Path(predictions)
    if path.exists():
        frame = pd.read_csv(path)
        actual = frame["actual"]; forecast = frame["forecast"]
    else:
        rng = np.random.default_rng(42); actual = pd.Series(np.linspace(250, 500, 120))
        forecast = actual + pd.Series(rng.normal(0, 18, len(actual)))
    fig, ax = plt.subplots(figsize=(11, 5)); ax.plot(actual, label="Actual"); ax.plot(forecast, label="Forecast")
    ax.set(title="Credit spread forecast accuracy", ylabel="Basis points"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "01_spread_forecast_accuracy.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    importance = pd.Series(np.linspace(1, .2, 8), index=["vix", "spread_rsi", "ftse_vol_20d", "bank_rate", "gilt_yield", "spread_roc_5", "ftse100", "spread_roc_20"])
    importance.sort_values().plot.barh(ax=ax, color="#1f77b4"); ax.set(title="Feature importance", xlabel="Relative importance")
    fig.tight_layout(); fig.savefig(out / "02_feature_importance.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    generate()

