"""Data ingestion for the credit-spread forecasting pipeline."""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests


def load_config(path: str | Path = "data_paths.txt") -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_boe_csv(path: str | Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=["date", column])
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", ""), errors="coerce")
    frame = frame.dropna().set_index("date").sort_index()
    if any(token in column.lower() for token in ("rate", "yield", "spread")):
        frame[column] *= 100
    return frame


def load_ftse_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, on_bad_lines="skip")
    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"].astype(str).str.replace(",", ""), errors="coerce")
    return frame.dropna(subset=["Date", "Price"]).set_index("Date")[["Price"]].rename(columns={"Price": "ftse100"}).sort_index()


def fetch_csv(url: str, date_column: str, value_column: str, scale: float = 1.0) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce") * scale
    return frame.dropna().set_index(date_column)[[value_column]].sort_index()


def assemble_dataset(config: Mapping[str, str]) -> pd.DataFrame:
    spread = fetch_csv(config["FRED_SPREAD_URL"], "DATE", "BAMLH0A0HYM2", 100).rename(columns={"BAMLH0A0HYM2": "spread"})
    bank = load_boe_csv(config["BOE_BANK_RATE_FILE"], "bank_rate")
    gilt = load_boe_csv(config["BOE_GILT_YIELD_FILE"], "gilt_yield")
    ftse = load_ftse_csv(config["FTSE_100_FILE_PATH"])
    vix = fetch_csv(config["CBOE_VIX_URL"], "DATE", "CLOSE").rename(columns={"CLOSE": "vix"})
    return spread.join([bank, gilt, ftse, vix], how="left").ffill().dropna()

