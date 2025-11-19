#!/usr/bin/env python3
"""Utility helpers used by the forecasting notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from .timeseries import build_forecast_index, infer_offset, select_series


@dataclass(frozen=True)
class DatasetSummary:
    """Basic information about the stacked quarterly dataset."""

    num_rows: int
    series_ids: list[str]
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class CountryStats:
    """Descriptive statistics computed per country."""

    series_id: str
    observations: int
    mean: float
    std: float
    coverage: float


@dataclass(frozen=True)
class CountryContext:
    """Prepared context for a single country ready to feed the runners."""

    dataframe: pd.DataFrame
    series_id: str
    offset: BaseOffset
    forecast_index: pd.DatetimeIndex


def load_quarterly_dataset(
    csv_path: Path,
    id_column: str = "country",
    timestamp_column: str = "timestamp",
    time_format: Optional[str] = None,
) -> tuple[pd.DataFrame, DatasetSummary]:
    """Load the stacked quarterly CSV and report the main metadata."""
    df = pd.read_csv(csv_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=time_format)
    ids = sorted(df[id_column].astype(str).unique())
    summary = DatasetSummary(
        num_rows=len(df),
        series_ids=ids,
        start=df[timestamp_column].min(),
        end=df[timestamp_column].max(),
    )
    return df, summary


def compute_country_stats(
    df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> list[CountryStats]:
    """Aggregate per-country statistics for a quick health check."""
    stats: list[CountryStats] = []
    for series_id, subset in df.groupby(id_column):
        values = subset[target_column]
        stats.append(
            CountryStats(
                series_id=str(series_id),
                observations=len(subset),
                mean=float(values.mean()),
                std=float(values.std()),
                coverage=float(values.notna().mean()),
            )
        )
    return sorted(stats, key=lambda item: item.series_id)


def prepare_country_context(
    df: pd.DataFrame,
    prediction_length: int,
    series_id: str | None = None,
    id_column: str = "country",
    timestamp_column: str = "timestamp",
    target_column: str = "investment",
    dropna: bool = True,
) -> CountryContext:
    """Select, clean, and augment the context for a single country."""
    context_df, resolved_id = select_series(
        df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
        series_id=series_id,
    )
    if dropna:
        context_df = context_df.dropna(subset=[target_column])
    context_df = context_df.sort_values(timestamp_column).reset_index(drop=True)
    offset = infer_offset(context_df[timestamp_column])
    forecast_index = build_forecast_index(
        context_df[timestamp_column],
        prediction_length,
        offset=offset,
    )
    return CountryContext(
        dataframe=context_df,
        series_id=resolved_id,
        offset=offset,
        forecast_index=forecast_index,
    )


def prepare_moirai_inputs(
    context_df: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    freq: str = "3M",
) -> tuple[np.ndarray, pd.Timestamp, str]:
    """Return the numpy target array and aligned start timestamp for Moirai."""
    values = context_df[target_column].to_numpy(dtype=np.float32)
    start_timestamp = pd.Timestamp(context_df[timestamp_column].iloc[0])
    if freq.upper() == "3M":
        # Align to the start of the month to match the expected 3M cadence.
        start_timestamp = start_timestamp.to_period("M").to_timestamp()
    return values, start_timestamp, freq
