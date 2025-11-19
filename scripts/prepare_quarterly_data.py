#!/usr/bin/env python3
"""Aggregate the monthly investment CSVs to quarterly frequency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import pandas as pd

# Columns needing quarterly averages vs sums. Columns missing from a dataset are ignored.
QUARTERLY_AVERAGE_COLUMNS = [
    "production_kg",
    "turnover_kg",
    "confidence_kg",
    "orders_kg",
    "exportorders_kg",
    "expectedprod_kg",
    "confidence_ind",
    "orders_ind",
    "exportorders_ind",
    "expectedprod_ind",
    "eei",
]

QUARTERLY_SUM_COLUMNS = [
    "imports_xea_kg",
    "imports_ea_kg",
    "imports_kg",
    "exports_xea_kg",
    "exports_ea_kg",
    "exports_kg",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the monthly investment CSVs into quarterly aggregates.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory/URI that holds the input CSV files (default: data).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="investment_*.csv",
        help="Glob pattern to select the files to process (default: investment_*.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where the quarterly CSVs will be written (default: data/processed).",
    )
    parser.add_argument(
        "--combined-name",
        type=str,
        default="investment_quarterly.csv",
        help="Filename for the stacked multi-country dataset (default: investment_quarterly.csv).",
    )
    parser.add_argument(
        "--interest-rate-csv",
        type=Path,
        default=Path("data/interest_rate.csv"),
        help="Optional CSV with observation_date/interest_rate/country columns to merge after aggregation.",
    )
    parser.add_argument(
        "--drop-redundant-columns",
        action="store_true",
        help="Drop *_kg columns when a *_ind counterpart exists before aggregation.",
    )
    return parser.parse_args(argv)


def is_s3_path(location: str | Path) -> bool:
    return str(location).startswith("s3://")


def list_input_files(input_dir: str, pattern: str) -> list[str | Path]:
    if is_s3_path(input_dir):
        try:
            import fsspec
        except ImportError as exc:
            raise RuntimeError("Reading from S3 requires the 'fsspec' and 's3fs' packages.") from exc
        fs, root = fsspec.core.url_to_fs(input_dir)
        normalized_root = root.rstrip("/")
        search_path = f"{normalized_root}/{pattern}"
        matches = fs.glob(search_path)
        if isinstance(fs.protocol, (list, tuple)):
            protocol = fs.protocol[0]
        else:
            protocol = fs.protocol
        return [f"{protocol}://{match}" for match in sorted(matches)]
    path = Path(input_dir)
    return sorted(path.glob(pattern))


def extract_path_stem(path: str | Path) -> str:
    text = str(path)
    if is_s3_path(text):
        parsed = urlparse(text)
        text = parsed.path
    return Path(text).stem


def load_monthly_frame(path: str | Path, drop_redundant: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Column 'date' is missing from {path}.")
    if drop_redundant:
        df = drop_redundant_unit_columns(df)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def drop_redundant_unit_columns(
    df: pd.DataFrame,
    keep_suffix: str = "_ind",
    drop_suffix: str = "_kg",
) -> pd.DataFrame:
    """Drop *_kg columns that have a *_ind counterpart."""
    drop_candidates: list[str] = []
    columns = set(df.columns)
    for column in df.columns:
        if column.endswith(drop_suffix):
            prefix = column[: -len(drop_suffix)]
            counterpart = f"{prefix}{keep_suffix}"
            if counterpart in columns:
                drop_candidates.append(column)
    if drop_candidates:
        df = df.drop(columns=drop_candidates)
    return df


def aggregate_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.set_index("date")
    # Baseline: keep the latest monthly observation for all columns.
    quarterly = work.resample("Q").last()

    def apply_custom_aggregation(columns: list[str], method: str) -> None:
        existing = [col for col in columns if col in work.columns]
        if not existing:
            return
        aggregated = work[existing].resample("Q").agg(method)
        quarterly[existing] = aggregated

    apply_custom_aggregation(QUARTERLY_AVERAGE_COLUMNS, "mean")
    apply_custom_aggregation(QUARTERLY_SUM_COLUMNS, "sum")

    quarterly.index.name = "quarter_end"
    quarterly = quarterly.reset_index()
    quarterly["timestamp"] = quarterly["quarter_end"].dt.to_period("Q").dt.to_timestamp(how="start")
    quarterly = quarterly.drop(columns=["quarter_end"])
    quarterly = quarterly.sort_values("timestamp").reset_index(drop=True)
    return quarterly


def load_interest_rates(path: Path | None) -> pd.DataFrame | None:
    """Load the optional per-country interest-rate series."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"⚠️  Interest-rate file '{path}' not found. Proceeding without it.")
        return None
    df = pd.read_csv(path)
    required = {"observation_date", "interest_rate", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Interest-rate CSV is missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["observation_date"], dayfirst=True)
    df = (
        df[["country", "timestamp", "interest_rate"]]
        .sort_values(["country", "timestamp"])
        .drop_duplicates(subset=["country", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return df


def merge_interest_rate(
    quarterly: pd.DataFrame,
    country: str,
    interest_rates: pd.DataFrame | None,
) -> pd.DataFrame:
    """Append the interest_rate column for the matching country if available."""
    if interest_rates is None:
        return quarterly
    country_rates = interest_rates[interest_rates["country"] == country]
    if country_rates.empty:
        return quarterly
    merged = quarterly.merge(
        country_rates[["timestamp", "interest_rate"]],
        on="timestamp",
        how="left",
    )
    return merged


def summarize_dataset(country: str, monthly: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    start = quarterly["timestamp"].min()
    end = quarterly["timestamp"].max()
    coverage = quarterly["investment"].notna().mean()
    print(f"{country}: {len(monthly):>4} monthly rows -> {len(quarterly):>3} quarters | "
          f"investment coverage {coverage:.1%} | {start.date()} -> {end.date()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_frames: list[pd.DataFrame] = []
    interest_rates = load_interest_rates(args.interest_rate_csv)

    files = list_input_files(args.input_dir, args.pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern '{args.pattern}' under {args.input_dir}.")

    for source_path in files:
        stem = extract_path_stem(source_path)
        country = stem.split("_")[-1]
        monthly = load_monthly_frame(source_path, drop_redundant=args.drop_redundant_columns)
        quarterly = aggregate_quarterly(monthly)
        quarterly = merge_interest_rate(quarterly, country, interest_rates)
        quarterly.insert(0, "country", country)
        combined_frames.append(quarterly)
        summarize_dataset(country, monthly, quarterly)
        per_country_path = output_dir / f"{stem}_quarterly.csv"
        quarterly.to_csv(per_country_path, index=False)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = combined.sort_values(["country", "timestamp"]).reset_index(drop=True)
    combined.to_csv(output_dir / args.combined_name, index=False)
    print(f"\nWrote {len(files)} per-country files and the combined dataset to {output_dir}.")
    print(f"Chronos target column: 'investment' | id column: 'country' | timestamp column: 'timestamp'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
