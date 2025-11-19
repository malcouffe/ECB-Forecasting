"""Reusable time-series helpers shared across notebooks and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def infer_offset(timestamps: pd.Series) -> pd.tseries.offsets.BaseOffset:
    """Infer the calendar offset directly from a timestamp series."""
    timestamps = pd.to_datetime(timestamps.sort_values().reset_index(drop=True))
    freq = pd.infer_freq(timestamps)
    if freq:
        return pd.tseries.frequencies.to_offset(freq)
    if len(timestamps) >= 2:
        delta = timestamps.iloc[-1] - timestamps.iloc[-2]
        return pd.tseries.frequencies.to_offset(delta)
    # Fallback to one day if we cannot infer anything else.
    return pd.tseries.frequencies.to_offset("D")


def offset_to_freq_str(offset: pd.tseries.offsets.BaseOffset) -> str:
    """Convert a pandas offset to a string code compatible with ListDataset."""
    for attr in ("freqstr", "rule_code", "name"):
        value = getattr(offset, attr, None)
        if isinstance(value, str) and value:
            return value
    try:
        derived = pd.tseries.frequencies.to_offset(offset)
    except Exception:
        return "D"
    for attr in ("freqstr", "rule_code", "name"):
        value = getattr(derived, attr, None)
        if isinstance(value, str) and value:
            return value
    return "D"


def build_forecast_index(
    timestamps: pd.Series,
    prediction_length: int,
    offset: pd.tseries.offsets.BaseOffset | None = None,
) -> pd.DatetimeIndex:
    """Return the DatetimeIndex corresponding to the future forecast horizon."""
    offset = offset or infer_offset(timestamps)
    start = timestamps.max() + offset
    return pd.date_range(start=start, periods=prediction_length, freq=offset)


def generate_context(
    length: int,
    freq: str,
    seed: int,
    series_id: str,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Create a simple synthetic context dataframe for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=length, freq=freq)
    signal = np.sin(np.linspace(0, 4 * np.pi, length))
    noise = rng.normal(0.0, 0.1, length)
    values = signal + noise
    return pd.DataFrame(
        {
            "id": series_id,
            "timestamp": dates,
            "target": values,
        }
    )


def load_context_from_csv(
    path: Path,
    id_column: str,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Load an existing CSV dataset and standardize the relevant columns."""
    df = pd.read_csv(path)
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found in {path}.")
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    for column in (id_column, target_column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {path}.")
    return df[[id_column, timestamp_column, target_column]].dropna()


def select_series(
    df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    series_id: str | None,
) -> tuple[pd.DataFrame, str]:
    """Extract a single time-series identified by `series_id`."""
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    all_ids = df[id_column].unique()
    if series_id is None:
        series_id = str(all_ids[0])
    if series_id not in df[id_column].values:
        raise ValueError(
            f"Series id '{series_id}' not found. Available ids: {sorted(all_ids)}"
        )
    subset = df[df[id_column] == series_id].sort_values(timestamp_column).reset_index(drop=True)
    subset = subset[[id_column, timestamp_column, target_column]]
    return subset, series_id


def quantile_column_name(prefix: str, quantile: float) -> str:
    """Return the canonical column name for a quantile level."""
    pct = int(round(quantile * 100))
    return f"{prefix}_p{pct:02d}"


__all__ = [
    "build_forecast_index",
    "generate_context",
    "infer_offset",
    "load_context_from_csv",
    "offset_to_freq_str",
    "quantile_column_name",
    "select_series",
]
