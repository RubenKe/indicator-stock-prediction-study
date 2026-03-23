from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .features import FEATURE_COLUMNS, REQUIRED_RAW_COLUMNS, build_feature_frame
from .types import DatasetSpec
from .utils import utc_now_iso


MANIFEST_FILE_NAME = "manifest.json"


def discover_raw_datasets(data_dir: Path) -> list[DatasetSpec]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    specs: list[DatasetSpec] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        if "_" not in csv_path.stem:
            continue
        symbol, interval = csv_path.stem.rsplit("_", 1)
        dataset_id = f"{symbol}_{interval}"
        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                symbol=symbol,
                interval=interval,
                raw_path=csv_path,
            )
        )
    return specs


def load_raw_ohlcv(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.raw_path, index_col=0, parse_dates=True)
    lower_cols = [str(c).lower() for c in df.columns]
    df.columns = lower_cols

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{spec.dataset_id}' missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    out = df[REQUIRED_RAW_COLUMNS].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _manifest_path(feature_cache_dir: Path) -> Path:
    return feature_cache_dir / MANIFEST_FILE_NAME


def load_feature_manifest(feature_cache_dir: Path) -> dict:
    manifest_path = _manifest_path(feature_cache_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_feature_cache(
    data_dir: Path,
    feature_cache_dir: Path,
    test_candles: int,
    force: bool = False,
) -> dict:
    specs = discover_raw_datasets(data_dir)
    if not specs:
        raise RuntimeError(f"No CSV datasets found in {data_dir}")

    if force and feature_cache_dir.exists():
        shutil.rmtree(feature_cache_dir)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_rows = []
    skipped_rows = []
    for spec in specs:
        raw_df = load_raw_ohlcv(spec)
        try:
            featured = build_feature_frame(
                raw_df=raw_df,
                dataset_id=spec.dataset_id,
                symbol=spec.symbol,
                interval=spec.interval,
                test_candles=test_candles,
            )
        except ValueError as exc:
            skipped_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "symbol": spec.symbol,
                    "interval": spec.interval,
                    "raw_file": spec.raw_path.name,
                    "raw_rows": int(len(raw_df)),
                    "reason": str(exc),
                }
            )
            continue

        feature_file = f"{spec.dataset_id}.parquet"
        feature_path = feature_cache_dir / feature_file
        featured.to_parquet(feature_path, index=True)

        dataset_manifest_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "symbol": spec.symbol,
                "interval": spec.interval,
                "raw_file": spec.raw_path.name,
                "feature_file": feature_file,
                "raw_rows": int(len(raw_df)),
                "feature_rows": int(len(featured)),
                "start": str(featured.index.min()),
                "end": str(featured.index.max()),
            }
        )

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "test_candles": int(test_candles),
        "required_raw_columns": REQUIRED_RAW_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "y",
        "return_column": "next_return",
        "datasets": dataset_manifest_rows,
        "skipped": skipped_rows,
    }
    manifest_path = _manifest_path(feature_cache_dir)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def ensure_feature_cache(
    data_dir: Path,
    feature_cache_dir: Path,
    test_candles: int,
    force: bool = False,
) -> dict:
    manifest_path = _manifest_path(feature_cache_dir)
    if force or not manifest_path.exists():
        return prepare_feature_cache(
            data_dir=data_dir,
            feature_cache_dir=feature_cache_dir,
            test_candles=test_candles,
            force=force,
        )
    return load_feature_manifest(feature_cache_dir)


def load_prepared_datasets(feature_cache_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = load_feature_manifest(feature_cache_dir)
    frames: dict[str, pd.DataFrame] = {}

    for item in manifest["datasets"]:
        dataset_id = item["dataset_id"]
        feature_file = item["feature_file"]
        path = feature_cache_dir / feature_file
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset for '{dataset_id}': expected {path}"
            )
        frame = pd.read_parquet(path)
        if "dataset_id" not in frame.columns:
            frame["dataset_id"] = dataset_id
        frames[dataset_id] = frame

    return frames
