import numpy as np
import pandas as pd

from src.features import build_features


def test_feature_builder_handles_constant_market_series():
    dates = pd.date_range("2020-01-01", periods=80)
    frame = pd.DataFrame(
        {"spread": 300.0, "bank_rate": 4.0, "gilt_yield": 3.0, "ftse100": 7000.0, "vix": 20.0},
        index=dates,
    )
    result = build_features(frame)
    assert not result.empty
    assert np.isfinite(result.to_numpy()).all()
