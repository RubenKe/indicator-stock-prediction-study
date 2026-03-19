from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    symbol: str
    interval: str
    raw_path: Path
    feature_path: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    test_candles: int
    threshold_long: float
    threshold_short: float
    cv_splits: int
    cv_scoring: str
    random_state: int
    feature_cache_dir: Path
    model_registry_dir: Path
    results_path: Path
    experiment_version: str
    commission: float


@dataclass
class ModelRunResult:
    summary_row: dict[str, Any]
    model_path: Path
    metadata_path: Path
    candle_path: Path
