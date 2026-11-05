"""Train a reproducible baseline model and persist it with joblib."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from .data_pipeline import assemble_dataset, load_config
from .features import build_features


def train(data, output: str | Path = "data/model.pkl"):
    target = data["spread"].shift(-1).dropna()
    features = data.loc[target.index].drop(columns="spread")
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    split = int(len(features) * 0.8)
    model.fit(features.iloc[:split], target.iloc[:split])
    print(f"holdout MAE: {mean_absolute_error(target.iloc[split:], model.predict(features.iloc[split:])):.3f}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(features.columns)}, output)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data_paths.txt")
    parser.add_argument("--output", default="data/model.pkl")
    args = parser.parse_args()
    train(build_features(assemble_dataset(load_config(args.config))), args.output)
