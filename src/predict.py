"""Score the latest feature row with a trained spread model."""
from __future__ import annotations

import argparse
import joblib
from .data_pipeline import assemble_dataset, load_config
from .features import build_features


def predict(data, model_path="data/model.pkl"):
    bundle = joblib.load(model_path)
    return bundle["model"].predict(data[bundle["features"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data_paths.txt")
    parser.add_argument("--model", default="data/model.pkl")
    args = parser.parse_args()
    values = predict(build_features(assemble_dataset(load_config(args.config))), args.model)
    print(f"next-period spread forecast: {values[-1]:.2f} bps")

