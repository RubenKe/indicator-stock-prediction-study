import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INTERVAL_TO_PERIODS_PER_YEAR = {
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "1d": 365,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_dumps_sorted(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(json_dumps_sorted(value).encode("utf-8")).hexdigest()
    return digest[:length]


def resolve_path(project_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_root / path


def annualization_factor(interval: str) -> int:
    if interval not in INTERVAL_TO_PERIODS_PER_YEAR:
        raise KeyError(
            f"Unsupported interval '{interval}' for annualization. "
            f"Supported: {sorted(INTERVAL_TO_PERIODS_PER_YEAR.keys())}"
        )
    return INTERVAL_TO_PERIODS_PER_YEAR[interval]
