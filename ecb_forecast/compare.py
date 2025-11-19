#!/usr/bin/env python3
"""
Compare Chronos-2 and Moirai-2 on the same time-series example.

The script reproduces the synthetic dataset that is used in `chronos2_exploration.ipynb`
and runs inference with both models so their quantile forecasts can be compared side by side.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from .timeseries import (
    build_forecast_index,
    generate_context,
    infer_offset,
    load_context_from_csv,
    offset_to_freq_str,
    quantile_column_name,
    select_series,
)


@dataclass
class ForecastResult:
    df: pd.DataFrame
    samples: np.ndarray | None
    elapsed_s: float


def parse_quantiles(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one quantile level must be provided.")
    quantiles = [float(p) for p in parts]
    for value in quantiles:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Invalid quantile level '{value}'. Each quantile must be between 0 and 1.")
    return sorted(set(quantiles))


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_torch_dtype(value: str) -> torch.dtype | str:
    value = value.lower()
    if value == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype '{value}'. Pick from {sorted(mapping)} or 'auto'.")
    return mapping[value]


class Chronos2Runner:
    def __init__(self, model_name: str, device_map: str, dtype: torch.dtype | str):
        init_kwargs: dict[str, object] = {"device_map": device_map}
        init_kwargs["torch_dtype"] = dtype
        self.pipeline = Chronos2Pipeline.from_pretrained(model_name, **init_kwargs)

    def run(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: Sequence[float],
        id_column: str,
        timestamp_column: str,
        target_column: str,
    ) -> ForecastResult:
        start = time.perf_counter()
        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels),
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target_column,
        )
        elapsed = time.perf_counter() - start
        subset = pred_df[pred_df[id_column] == context_df[id_column].iloc[0]].copy()
        rename_map = {}
        for column in subset.columns:
            try:
                quantile = float(column)
            except ValueError:
                continue
            rename_map[column] = quantile_column_name("chronos", quantile)
        if "mean" in subset.columns:
            rename_map["mean"] = "chronos_mean"
        subset = subset.rename(columns=rename_map)
        keep_columns = [timestamp_column] + sorted(
            [name for name in rename_map.values() if name.startswith("chronos_")],
            key=lambda name: (name != "chronos_mean", name),
        )
        cleaned = subset[keep_columns].reset_index(drop=True)
        return ForecastResult(df=cleaned, samples=None, elapsed_s=elapsed)


class Moirai2Runner:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | str = "auto",
        context_length: int = 1680,
        batch_size: int = 1,
        device_map: str | None = None,
    ):
        if device is None:
            if device_map is not None:
                device = device_map
            else:
                device = resolve_device("auto")
        torch_dtype = None if dtype == "auto" else dtype
        self.device = torch.device(device)
        sanitized_dtype = self._sanitize_dtype(torch_dtype)
        # Some safetensors checkpoints don't support direct loading onto MPS,
        # so load on CPU first and move afterwards.
        load_location = "cpu"
        self.module = Moirai2Module.from_pretrained(
            model_name,
            map_location=load_location,
        )
        if sanitized_dtype is not None:
            self.module.to(self.device, dtype=sanitized_dtype)
        else:
            self.module.to(self.device)
        self.module.eval()
        safe_context = max(1, context_length)
        self.forecast = Moirai2Forecast(
            module=self.module,
            prediction_length=1,
            context_length=safe_context,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.max_context_length = safe_context
        self.batch_size = max(1, batch_size)
        self._dtype = sanitized_dtype

    def _build_dataset(
        self,
        values: Sequence[float],
        start_timestamp: pd.Timestamp,
        freq: str,
    ) -> ListDataset:
        return ListDataset(
            [
                {
                    "start": pd.to_datetime(start_timestamp),
                    "target": np.asarray(values, dtype=np.float32),
                }
            ],
            freq=freq,
        )

    @torch.inference_mode()
    def run(
        self,
        values: Sequence[float],
        prediction_length: int,
        quantile_levels: Sequence[float],
        forecast_index: pd.DatetimeIndex,
        start_timestamp: pd.Timestamp,
        freq: str,
    ) -> ForecastResult:
        dataset = self._build_dataset(values, start_timestamp, freq)
        effective_context = min(self.max_context_length, len(values))
        with self.forecast.hparams_context(
            prediction_length=prediction_length,
            context_length=effective_context,
        ):
            predictor = self.forecast.create_predictor(
                batch_size=self.batch_size,
                device=self.device,
            )
            start_time = time.perf_counter()
            forecasts = list(predictor.predict(dataset))
        elapsed = time.perf_counter() - start_time
        if not forecasts:
            raise RuntimeError("Moirai-2 predictor did not return any forecasts.")
        forecast = forecasts[0]
        summary = {
            quantile_column_name("moirai", q): np.asarray(forecast.quantile(q))
            for q in quantile_levels
        }
        summary["moirai_mean"] = np.asarray(forecast.mean)
        summary_df = pd.DataFrame(
            {
                "timestamp": forecast_index,
                **summary,
            }
        )
        return ForecastResult(df=summary_df, samples=None, elapsed_s=elapsed)

    @staticmethod
    def _sanitize_dtype(dtype: torch.dtype | None) -> torch.dtype | None:
        if dtype is None or dtype == torch.float32:
            return dtype
        warnings.warn(
            f"Moirai-2 runner expects float32 inputs; forcing dtype to torch.float32 "
            f"instead of requested {dtype}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.float32


def merge_results(
    chronos: ForecastResult,
    moirai: ForecastResult,
    timestamp_column: str,
) -> pd.DataFrame:
    merged = pd.merge(
        chronos.df,
        moirai.df,
        on=timestamp_column,
        how="inner",
        suffixes=("_chronos", "_moirai"),
    ).sort_values(timestamp_column)
    chrono_median = next(
        (col for col in merged.columns if col.startswith("chronos_p50")), None
    )
    moirai_median = next(
        (col for col in merged.columns if col.startswith("moirai_p50")), None
    )
    if chrono_median and moirai_median:
        merged["delta_p50"] = merged[chrono_median] - merged[moirai_median]
    return merged.reset_index(drop=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Chronos-2 and Moirai-2 forecasts on the same time series."
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=14,
        help="Number of future steps to forecast.",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.1,0.5,0.9",
        help="Comma-separated list of quantile levels to compute.",
    )
    parser.add_argument(
        "--chronos-model",
        type=str,
        default="amazon/chronos-2",
        help="Chronos-2 model name on Hugging Face Hub.",
    )
    parser.add_argument(
        "--moirai-model",
        type=str,
        default="Salesforce/moirai-2.0-R-base",
        help="Moirai-2 model name on Hugging Face Hub.",
    )
    parser.add_argument(
        "--chronos-device",
        type=str,
        default="auto",
        help="Device map for Chronos-2 (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--moirai-device",
        type=str,
        default="auto",
        help="Device map for Moirai-2 (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--chronos-dtype",
        type=str,
        default="auto",
        help="Torch dtype for Chronos-2 (auto, float32, float16, bfloat16).",
    )
    parser.add_argument(
        "--moirai-dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for Moirai-2 (auto, float32, float16, bfloat16).",
    )
    parser.add_argument(
        "--moirai-context-length",
        type=int,
        default=1680,
        help="Maximum history length provided to Moirai-2 (clipped to the available observations).",
    )
    parser.add_argument(
        "--moirai-batch-size",
        type=int,
        default=1,
        help="Batch size to use for the Moirai-2 predictor.",
    )
    parser.add_argument(
        "--context-csv",
        type=Path,
        help="Optional CSV file containing id/timestamp/target columns. If omitted, a synthetic sine wave is used.",
    )
    parser.add_argument(
        "--series-id",
        type=str,
        help="Identifier of the series to forecast (required when the context CSV holds multiple series).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Name of the identifier column.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp",
        help="Name of the timestamp column.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--synthetic-length",
        type=int,
        default=100,
        help="Length of the synthetic context when no CSV is provided.",
    )
    parser.add_argument(
        "--synthetic-freq",
        type=str,
        default="D",
        help="Frequency string for the synthetic context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to generate the synthetic example.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the merged comparison as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    quantiles = parse_quantiles(args.quantiles)
    if args.context_csv:
        context_df = load_context_from_csv(
            args.context_csv,
            args.id_column,
            args.timestamp_column,
            args.target_column,
        )
    else:
        context_df = generate_context(
            length=args.synthetic_length,
            freq=args.synthetic_freq,
            seed=args.seed,
            series_id=args.series_id or "series_1",
        )
    context_df, active_series_id = select_series(
        context_df,
        args.id_column,
        args.timestamp_column,
        args.target_column,
        args.series_id,
    )
    print(f"Using series id '{active_series_id}' with {len(context_df)} observations.")
    forecast_horizon = args.prediction_length
    context_offset = infer_offset(context_df[args.timestamp_column])
    forecast_index = build_forecast_index(
        context_df[args.timestamp_column], forecast_horizon, offset=context_offset
    )
    freq_str = offset_to_freq_str(context_offset)
    start_timestamp = pd.to_datetime(context_df[args.timestamp_column].iloc[0])

    chronos_device = resolve_device(args.chronos_device)
    moirai_device = resolve_device(args.moirai_device)
    chronos_dtype = parse_torch_dtype(args.chronos_dtype)
    moirai_dtype = parse_torch_dtype(args.moirai_dtype)

    chronos_runner = Chronos2Runner(args.chronos_model, chronos_device, chronos_dtype)
    chronos_result = chronos_runner.run(
        context_df=context_df,
        prediction_length=forecast_horizon,
        quantile_levels=quantiles,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
    )
    print(f"Chronos-2 inference completed in {chronos_result.elapsed_s:.2f}s on {chronos_device}.")

    moirai_runner = Moirai2Runner(
        args.moirai_model,
        moirai_device,
        moirai_dtype,
        context_length=args.moirai_context_length,
        batch_size=args.moirai_batch_size,
    )
    context_values = context_df[args.target_column].to_numpy(dtype=np.float32)
    moirai_result = moirai_runner.run(
        values=context_values,
        prediction_length=forecast_horizon,
        quantile_levels=quantiles,
        forecast_index=forecast_index,
        start_timestamp=start_timestamp,
        freq=freq_str,
    )
    print(f"Moirai-2 inference completed in {moirai_result.elapsed_s:.2f}s on {moirai_device}.")

    merged = merge_results(chronos_result, moirai_result, args.timestamp_column)
    print("\nHead of the merged comparison:")
    print(merged.head())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output, index=False)
        print(f"\nSaved merged forecasts to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
