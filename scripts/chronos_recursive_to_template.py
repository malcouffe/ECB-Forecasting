#!/usr/bin/env python3
"""Run Chronos-2 recursive forecasts and export them to the TeamXX template."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecb_forecast.compare import Chronos2Runner, parse_quantiles, parse_torch_dtype
from ecb_forecast.datasets import prepare_country_context


@dataclass(frozen=True)
class TargetConfig:
    """Transformer configuration describing how to post-process predictions."""

    target_column: str
    actual_column: str
    requires_qoq: bool
    level_column: str | None = None

    @property
    def needs_level_history(self) -> bool:
        return self.requires_qoq and self.level_column is not None


DEFAULT_TARGETS = ["investment", "gdp_qoq", "interest_rate", "investment_qoq"]
TEMPLATE_COLUMNS = [
    "cutoff",
    "oos_date",
    "y_true",
    "y_pred",
    "quantile_0.1",
    "quantile_0.2",
    "quantile_0.3",
    "quantile_0.4",
    "quantile_0.5",
    "quantile_0.6",
    "quantile_0.7",
    "quantile_0.8",
    "quantile_0.9",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce recursive Chronos-2 forecasts and export them in the TeamXX template.",
    )
    parser.add_argument("--country", type=str, default="DE", help="Country identifier to forecast (default: DE).")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target columns to forecast (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the quarterly CSV (defaults to data/processed/investment_<country>_quarterly.csv "
        "and falls back to data/processed/investment_quarterly.csv).",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=None,
        help="Excel workbook that will store the formatted sheets (default: auto-generated timestamped file under results/).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Recursive prediction horizon per origin (default: 4 quarters).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-10-01",
        help="First cutoff (inclusive) to launch recursive forecasts (default: 2020-10-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-01-01",
        help="Last cutoff (inclusive) to launch recursive forecasts (default: 2025-01-01).",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.1,0.5,0.9",
        help="Comma-separated list of Chronos quantiles to request (default: 0.1,0.5,0.9).",
    )
    parser.add_argument(
        "--chronos-model",
        type=str,
        default="amazon/chronos-2",
        help="Chronos-2 model name on the Hugging Face Hub (default: amazon/chronos-2).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to Chronos-2 (default: auto).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype for Chronos-2 weights (default: auto).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="country",
        help="Identifier column in the dataset (default: country).",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp",
        help="Timestamp column name in the dataset (default: timestamp).",
    )
    return parser.parse_args(argv)


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset:
        return args.dataset
    per_country = PROJECT_ROOT / "data" / "processed" / f"investment_{args.country}_quarterly.csv"
    if per_country.exists():
        return per_country
    combined = PROJECT_ROOT / "data" / "processed" / "investment_quarterly.csv"
    if combined.exists():
        return combined
    raise FileNotFoundError(
        "Could not find a quarterly dataset. Provide --dataset explicitly or place the processed CSV "
        "under data/processed."
    )


def load_country_frame(
    csv_path: Path,
    country: str,
    id_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found in {csv_path}.")
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    if id_column not in df.columns:
        df[id_column] = country
    subset = df[df[id_column] == country].copy()
    if subset.empty:
        available = sorted(df[id_column].astype(str).unique())
        raise ValueError(f"No rows for country '{country}' in {csv_path}. Available ids: {available}")
    subset = subset.sort_values(timestamp_column).reset_index(drop=True)
    return subset


def resolve_target_config(target: str) -> TargetConfig:
    requires_qoq = target == "investment"
    actual_column = f"{target}_qoq" if requires_qoq else target
    level_column = target if requires_qoq else None
    return TargetConfig(
        target_column=target,
        actual_column=actual_column,
        requires_qoq=requires_qoq,
        level_column=level_column,
    )


def clamp_origin_range(
    subset: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    timestamp_column: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    data_start = subset[timestamp_column].min()
    data_end = subset[timestamp_column].max()
    start = max(data_start, start_date)
    end = min(data_end, end_date)
    if start > end:
        raise ValueError(f"Date range [{start_date}, {end_date}] is outside the available data [{data_start}, {data_end}].")
    return start, end


def run_recursive_forecasts(
    runner: Chronos2Runner,
    subset: pd.DataFrame,
    country: str,
    config: TargetConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: int,
    quantile_levels: Sequence[float],
    id_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    work = subset.dropna(subset=[config.target_column]).copy()
    if work.empty:
        raise ValueError(f"No observations for target '{config.target_column}'.")
    origins = sorted(
        ts
        for ts in work[timestamp_column].unique()
        if start_date <= ts <= end_date
    )
    if not origins:
        raise ValueError(f"No forecast origins fall between {start_date.date()} and {end_date.date()}.")

    results: list[pd.DataFrame] = []
    print(
        f"\n🔄 Chronos-2 recursive forecasting for '{config.target_column}' ({len(origins)} origins, horizon={horizon})"
    )

    for idx, origin in enumerate(origins, start=1):
        context = work[work[timestamp_column] <= origin].copy()
        if context[config.target_column].notna().sum() < max(4, horizon):
            print(f"  Skipping origin {origin.date()} (insufficient history).")
            continue
        country_context = prepare_country_context(
            context,
            prediction_length=horizon,
            series_id=country,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=config.target_column,
        )
        forecast = runner.run(
            context_df=country_context.dataframe,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=config.target_column,
        )
        frame = forecast.df.copy()
        frame["origin_date"] = origin
        frame["horizon_step"] = range(1, horizon + 1)
        frame["model"] = "Chronos-2"
        results.append(frame)

        if idx % 5 == 0 or idx == len(origins):
            print(f"  Processed {idx}/{len(origins)} origins...")

    if not results:
        raise RuntimeError("Chronos did not produce any forecast rows.")
    combined = pd.concat(results, ignore_index=True)
    combined[timestamp_column] = pd.to_datetime(combined[timestamp_column])
    combined["origin_date"] = pd.to_datetime(combined["origin_date"])
    return combined


def interpolate_quantiles(p10: float, p50: float, p90: float) -> tuple[float, float, float, float, float, float]:
    q02 = p10 + (p50 - p10) * 0.25
    q03 = p10 + (p50 - p10) * 0.5
    q04 = p10 + (p50 - p10) * 0.75
    q06 = p50 + (p90 - p50) * 0.25
    q07 = p50 + (p90 - p50) * 0.5
    q08 = p50 + (p90 - p50) * 0.75
    return q02, q03, q04, q06, q07, q08


def compute_qoq_percentage(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def lookup_previous_level(
    history: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    origin: pd.Timestamp,
) -> float | None:
    subset = history[history[timestamp_column] <= origin]
    subset = subset.dropna(subset=[value_column])
    if subset.empty:
        return None
    return float(subset.iloc[-1][value_column])


def format_forecasts_for_template(
    forecasts: pd.DataFrame,
    history: pd.DataFrame,
    config: TargetConfig,
    timestamp_column: str,
) -> pd.DataFrame:
    if config.actual_column not in history.columns:
        raise ValueError(f"Column '{config.actual_column}' missing from the dataset.")
    actual_map = (
        history[[timestamp_column, config.actual_column]]
        .dropna(subset=[config.actual_column])
        .drop_duplicates(subset=[timestamp_column], keep="last")
        .set_index(timestamp_column)[config.actual_column]
    )
    template_rows: list[dict[str, object]] = []
    forecasts = forecasts.sort_values(["origin_date", "horizon_step", timestamp_column]).reset_index(drop=True)

    level_history = None
    if config.needs_level_history:
        if config.level_column not in history.columns:
            raise ValueError(f"Column '{config.level_column}' missing from the dataset.")
        level_history = history[[timestamp_column, config.level_column]].copy()

    for origin, group in forecasts.groupby("origin_date"):
        prev_value = None
        if config.needs_level_history:
            prev_value = lookup_previous_level(level_history, timestamp_column, config.level_column, origin)
            if prev_value is None:
                print(f"  Warning: missing baseline level for origin {origin.date()}, skipping these horizons.")
                continue

        for _, row in group.sort_values("horizon_step").iterrows():
            base_p10 = row.get("chronos_p10")
            base_p50 = row.get("chronos_p50")
            base_p90 = row.get("chronos_p90")
            if config.requires_qoq:
                p10 = compute_qoq_percentage(float(base_p10), prev_value)
                p50 = compute_qoq_percentage(float(base_p50), prev_value)
                p90 = compute_qoq_percentage(float(base_p90), prev_value)
            else:
                p10 = float(base_p10) if base_p10 is not None else None
                p50 = float(base_p50) if base_p50 is not None else None
                p90 = float(base_p90) if base_p90 is not None else None

            if config.requires_qoq:
                prev_value = None if base_p50 is None else float(base_p50)

            if p10 is None or p50 is None or p90 is None:
                q02 = q03 = q04 = q06 = q07 = q08 = None
            else:
                q02, q03, q04, q06, q07, q08 = interpolate_quantiles(p10, p50, p90)

            ts = row[timestamp_column]
            actual = actual_map.get(ts, np.nan)

            template_rows.append(
                {
                    "cutoff": pd.Timestamp(origin),
                    "oos_date": pd.Timestamp(ts),
                    "y_true": actual if pd.notna(actual) else None,
                    "y_pred": p50,
                    "quantile_0.1": p10,
                    "quantile_0.2": q02,
                    "quantile_0.3": q03,
                    "quantile_0.4": q04,
                    "quantile_0.5": p50,
                    "quantile_0.6": q06,
                    "quantile_0.7": q07,
                    "quantile_0.8": q08,
                    "quantile_0.9": p90,
                }
            )

    output = pd.DataFrame(template_rows)
    if output.empty:
        raise RuntimeError("No template rows were generated.")
    output["cutoff"] = output["cutoff"].apply(align_to_quarter_end)
    output["oos_date"] = output["oos_date"].apply(align_to_quarter_end)
    return output[TEMPLATE_COLUMNS]


def align_to_quarter_end(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return value
    ts = pd.Timestamp(value)
    if ts.is_quarter_start:
        return (ts - pd.offsets.Day(1)).normalize()
    return ts.normalize()


def save_excel_sheets(sheets: dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"\n✅ Saved {len(sheets)} sheet(s) to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_xlsx is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_xlsx = PROJECT_ROOT / "results" / f"TeamXX_recursive_chronos_predictions_{timestamp}.xlsx"
    else:
        output_xlsx = args.output_xlsx
    dataset_path = resolve_dataset_path(args)
    print(f"📁 Using dataset: {dataset_path}")
    subset = load_country_frame(dataset_path, args.country, args.id_column, args.timestamp_column)

    quantiles = parse_quantiles(args.quantiles)
    dtype = parse_torch_dtype(args.torch_dtype)
    runner = Chronos2Runner(args.chronos_model, args.device_map, dtype)

    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    start_date, end_date = clamp_origin_range(subset, start_date, end_date, args.timestamp_column)

    sheet_frames: dict[str, pd.DataFrame] = {}
    for target in args.targets:
        config = resolve_target_config(target)
        missing_columns = {config.target_column}
        if config.requires_qoq:
            missing_columns.add(config.actual_column)
        for column in missing_columns:
            if column not in subset.columns:
                raise ValueError(f"Column '{column}' required for '{target}' is missing from the dataset.")

        forecasts = run_recursive_forecasts(
            runner=runner,
            subset=subset,
            country=args.country,
            config=config,
            start_date=start_date,
            end_date=end_date,
            horizon=args.horizon,
            quantile_levels=quantiles,
            id_column=args.id_column,
            timestamp_column=args.timestamp_column,
        )
        template = format_forecasts_for_template(
            forecasts=forecasts,
            history=subset,
            config=config,
            timestamp_column=args.timestamp_column,
        )
        sheet_name = f"{target}_{args.country}"
        sheet_frames[sheet_name] = template
        print(
            f"  → Sheet '{sheet_name}' ready with {len(template)} rows covering "
            f"{template['oos_date'].min()} → {template['oos_date'].max()}"
        )

    save_excel_sheets(sheet_frames, output_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
