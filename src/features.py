"""Leak-free, time-series features for spread forecasting."""
from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ftse_vol_20d"] = data["ftse100"].pct_change().rolling(20, min_periods=2).std() * np.sqrt(252) * 100
    data["spread_rsi"] = calculate_rsi(data["spread"])
    data["vix_pct_from_ma10"] = (data["vix"] - data["vix"].rolling(10).mean()) / data["vix"].rolling(10).mean() * 100
    data["ftse_vol_macd"] = data["ftse_vol_20d"] - data["ftse_vol_20d"].rolling(50).mean()
    data["spread_roc_5"] = data["spread"].pct_change(5) * 100
    data["spread_roc_20"] = data["spread"].pct_change(20) * 100
    data["vix_roc_5"] = data["vix"].pct_change(5) * 100
    spread_z = (data["spread"] - data["spread"].rolling(20).mean()) / data["spread"].rolling(20).std()
    vix_z = (data["vix"] - data["vix"].rolling(20).mean()) / data["vix"].rolling(20).std()
    data["spread_vix_divergence"] = spread_z - vix_z
    for column in ("spread_rsi", "vix_pct_from_ma10", "spread_roc_5"):
        for lag in (1, 5):
            data[f"{column}_lag_{lag}"] = data[column].shift(lag)
    return data.replace([np.inf, -np.inf], np.nan).fillna(0)
