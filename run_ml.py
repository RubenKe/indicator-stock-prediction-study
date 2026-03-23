import argparse
import json
from pathlib import Path

import yaml

from ml.datasets import ensure_feature_cache, load_prepared_datasets
from ml.features import FEATURE_COLUMNS
from ml.models import parse_model_list, validate_profile
from ml.persistence import (
    append_result_rows,
    bootstrap_results_file,
    load_existing_experiment_keys,
    save_model_artifacts,
    save_run_summary,
)
from ml.train_eval import (
    build_train_test_frames,
    evaluate_model_on_test,
    tune_model_with_group_kfold,
)
from ml.types import RunConfig
from ml.utils import resolve_path, stable_hash, utc_now_compact, utc_now_iso


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANALYSIS_ML_DIR = PROJECT_ROOT / "analysis" / "results" / "ml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-dataset-out ML experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare feature cache from data/raw into data/features.",
    )
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        help="Run ML for one test dataset against all other datasets.",
    )
    run_parser.add_argument("--test-dataset", required=True)
    run_parser.add_argument(
        "--models",
        default="logistic,random_forest,gradient_boosting",
    )
    run_parser.add_argument("--profile", default="standard")
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--n-jobs", type=int, default=-1)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run ML for all datasets in leave-one-dataset-out mode.",
    )
    run_all_parser.add_argument(
        "--models",
        default="logistic,random_forest,gradient_boosting",
    )
    run_all_parser.add_argument("--profile", default="standard")
    run_all_parser.add_argument("--max-tests", type=int, default=0)
    run_all_parser.add_argument("--force", action="store_true")
    run_all_parser.add_argument("--seed", type=int, default=None)
    run_all_parser.add_argument("--n-jobs", type=int, default=-1)

    return parser.parse_args()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_config(cfg: dict, seed_override: int | None = None) -> RunConfig:
    ml_cfg = cfg.get("ml", {})
    random_state = (
        int(seed_override) if seed_override is not None else int(ml_cfg.get("random_state", 42))
    )
    return RunConfig(
        test_candles=int(ml_cfg.get("test_candles", 1250)),
        threshold_long=float(ml_cfg.get("threshold_long", 0.55)),
        threshold_short=float(ml_cfg.get("threshold_short", 0.45)),
        cv_splits=int(ml_cfg.get("cv_splits", 5)),
        cv_scoring=str(ml_cfg.get("cv_scoring", "roc_auc")),
        random_state=random_state,
        feature_cache_dir=resolve_path(
            PROJECT_ROOT, ml_cfg.get("feature_cache_dir", "data/features")
        ),
        model_registry_dir=resolve_path(
            PROJECT_ROOT, ml_cfg.get("model_registry_dir", "models/ml_registry")
        ),
        results_path=resolve_path(
            PROJECT_ROOT, ml_cfg.get("results_path", "database/ml_results.parquet")
        ),
        experiment_version=str(ml_cfg.get("experiment_version", "loo_ml_v1")),
        commission=float(cfg.get("commission", 0.0)),
        slippage=float(cfg.get("slippage", 0.0)),
    )


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _make_run_id(seed: int, profile: str, model_names: list[str]) -> str:
    timestamp = utc_now_compact()
    payload = {
        "seed": seed,
        "profile": profile,
        "models": sorted(model_names),
        "timestamp": timestamp,
    }
    models_tag = "-".join(sorted(model_names))
    hash_suffix = stable_hash(payload, length=8)
    return f"ml_{timestamp}_models-{models_tag}_seed-{seed}_{hash_suffix}"


def prepare_cache(run_cfg: RunConfig, force: bool, exclude_symbols: set[str] | None = None) -> dict:
    manifest = ensure_feature_cache(
        data_dir=DATA_RAW_DIR,
        feature_cache_dir=run_cfg.feature_cache_dir,
        test_candles=run_cfg.test_candles,
        force=force,
        exclude_symbols=exclude_symbols,
    )
    print(
        f"Prepared feature cache at {run_cfg.feature_cache_dir} "
        f"for {len(manifest['datasets'])} datasets."
    )
    return manifest


def _run_for_test_dataset(
    run_id: str,
    test_dataset: str,
    model_names: list[str],
    profile: str,
    force: bool,
    seed: int,
    n_jobs: int,
    run_cfg: RunConfig,
    dataset_frames: dict,
    existing_keys: set[str],
) -> list[dict]:
    train_df, test_df, train_ids = build_train_test_frames(
        dataset_frames=dataset_frames,
        test_dataset=test_dataset,
    )

    rows: list[dict] = []
    symbol = str(test_df["symbol"].iloc[0])
    interval = str(test_df["interval"].iloc[0])

    for model_name in model_names:
        experiment_payload = {
            "experiment_version": run_cfg.experiment_version,
            "test_dataset": test_dataset,
            "model_name": model_name,
            "profile": profile,
            "seed": seed,
            "threshold_long": run_cfg.threshold_long,
            "threshold_short": run_cfg.threshold_short,
            "commission": run_cfg.commission,
            "slippage": run_cfg.slippage,
            "train_datasets": sorted(train_ids),
            "test_candles": run_cfg.test_candles,
        }
        experiment_key = stable_hash(experiment_payload, length=24)
        if not force and experiment_key in existing_keys:
            print(f"Skipping existing experiment: {test_dataset} / {model_name}")
            continue

        print(f"Training {model_name} | test={test_dataset}")
        tuned = tune_model_with_group_kfold(
            model_name=model_name,
            profile=profile,
            train_df=train_df,
            run_cfg=run_cfg,
            seed=seed,
            n_jobs=n_jobs,
        )
        estimator = tuned["estimator"]
        candle_df, metrics = evaluate_model_on_test(
            estimator=estimator,
            test_df=test_df,
            run_cfg=run_cfg,
        )

        metadata = {
            "run_id": run_id,
            "created_at_utc": utc_now_iso(),
            "experiment_version": run_cfg.experiment_version,
            "experiment_key": experiment_key,
            "test_dataset": test_dataset,
            "symbol": symbol,
            "interval": interval,
            "model_name": model_name,
            "profile": profile,
            "seed": seed,
            "best_params": tuned["best_params"],
            "cv_best_roc_auc": tuned["cv_best_roc_auc"],
            "cv_best_accuracy": tuned["cv_best_accuracy"],
            "cv_splits": tuned["cv_splits"],
            "n_train_rows": tuned["n_train_rows"],
            "n_test_rows": int(len(test_df)),
            "train_datasets": train_ids,
            "feature_columns": FEATURE_COLUMNS,
            "threshold_long": run_cfg.threshold_long,
            "threshold_short": run_cfg.threshold_short,
            "commission": run_cfg.commission,
            "slippage": run_cfg.slippage,
            "metrics": metrics,
        }

        paths = save_model_artifacts(
            run_id=run_id,
            test_dataset=test_dataset,
            model_name=model_name,
            estimator=estimator,
            candle_df=candle_df,
            metadata=metadata,
            model_registry_dir=run_cfg.model_registry_dir,
            analysis_ml_dir=ANALYSIS_ML_DIR,
        )

        row = {
            "run_id": run_id,
            "created_at_utc": utc_now_iso(),
            "experiment_key": experiment_key,
            "experiment_version": run_cfg.experiment_version,
            "test_dataset": test_dataset,
            "symbol": symbol,
            "interval": interval,
            "model_name": model_name,
            "profile": profile,
            "seed": seed,
            "n_train_datasets": len(train_ids),
            "train_datasets_json": json.dumps(train_ids, sort_keys=True),
            "n_train_rows": tuned["n_train_rows"],
            "n_test_rows": int(len(test_df)),
            "cv_splits": tuned["cv_splits"],
            "cv_best_roc_auc": tuned["cv_best_roc_auc"],
            "cv_best_accuracy": tuned["cv_best_accuracy"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_accuracy": metrics["test_accuracy"],
            "best_params_json": json.dumps(tuned["best_params"], sort_keys=True),
            "threshold_long": run_cfg.threshold_long,
            "threshold_short": run_cfg.threshold_short,
            "commission": run_cfg.commission,
            "slippage": run_cfg.slippage,
            "total_return_pct": metrics["total_return_pct"],
            "buy_hold_return_pct": metrics["buy_hold_return_pct"],
            "excess_return_pct": metrics["excess_return_pct"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "annualized_return_pct": metrics["annualized_return_pct"],
            "num_trades": metrics["num_trades"],
            "signal_coverage": metrics["signal_coverage"],
            "feature_columns_json": json.dumps(FEATURE_COLUMNS),
            "model_path": _relative(paths["model_path"]),
            "metadata_path": _relative(paths["metadata_path"]),
            "candle_path": _relative(paths["candle_path"]),
            "summary_path": "",
        }
        rows.append(row)
        existing_keys.add(experiment_key)
    return rows


def cmd_prepare(args):
    cfg = load_config()
    run_cfg = build_run_config(cfg)
    exclude_symbols = set(cfg.get("crypto", []))
    prepare_cache(run_cfg, force=args.force, exclude_symbols=exclude_symbols)


def _run_workflow(
    run_cfg: RunConfig,
    test_datasets: list[str],
    model_names: list[str],
    profile: str,
    force: bool,
    seed: int,
    n_jobs: int,
) -> None:
    dataset_frames = load_prepared_datasets(run_cfg.feature_cache_dir)
    available = sorted(dataset_frames.keys())

    for ds in test_datasets:
        if ds not in dataset_frames:
            raise KeyError(f"Test dataset '{ds}' not found. Available: {available}")

    bootstrap_results_file(run_cfg.results_path)
    existing_keys = set() if force else load_existing_experiment_keys(run_cfg.results_path)

    run_id = _make_run_id(seed=seed, profile=profile, model_names=model_names)
    print(f"Run ID: {run_id}")

    all_rows: list[dict] = []
    for test_dataset in test_datasets:
        rows = _run_for_test_dataset(
            run_id=run_id,
            test_dataset=test_dataset,
            model_names=model_names,
            profile=profile,
            force=force,
            seed=seed,
            n_jobs=n_jobs,
            run_cfg=run_cfg,
            dataset_frames=dataset_frames,
            existing_keys=existing_keys,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No new model runs were produced.")
        return

    summary_path = save_run_summary(run_id=run_id, summary_rows=all_rows, analysis_ml_dir=ANALYSIS_ML_DIR)
    summary_path_rel = _relative(summary_path)
    for row in all_rows:
        row["summary_path"] = summary_path_rel

    written = append_result_rows(
        results_path=run_cfg.results_path,
        rows=all_rows,
        dedupe_on_experiment_key=not force,
    )
    print(f"Wrote {written} rows to {run_cfg.results_path}")
    print(f"Summary: {summary_path}")


def cmd_run(args):
    cfg = load_config()
    run_cfg = build_run_config(cfg, seed_override=args.seed)
    model_names = parse_model_list(args.models)
    profile = validate_profile(args.profile)
    exclude_symbols = set(cfg.get("crypto", []))
    prepare_cache(run_cfg, force=False, exclude_symbols=exclude_symbols)

    _run_workflow(
        run_cfg=run_cfg,
        test_datasets=[args.test_dataset],
        model_names=model_names,
        profile=profile,
        force=args.force,
        seed=run_cfg.random_state,
        n_jobs=args.n_jobs,
    )


def cmd_run_all(args):
    cfg = load_config()
    run_cfg = build_run_config(cfg, seed_override=args.seed)
    model_names = parse_model_list(args.models)
    profile = validate_profile(args.profile)

    exclude_symbols = set(cfg.get("crypto", []))
    manifest = prepare_cache(run_cfg, force=False, exclude_symbols=exclude_symbols)
    test_datasets = sorted([d["dataset_id"] for d in manifest["datasets"]])
    if args.max_tests > 0:
        test_datasets = test_datasets[: args.max_tests]

    _run_workflow(
        run_cfg=run_cfg,
        test_datasets=test_datasets,
        model_names=model_names,
        profile=profile,
        force=args.force,
        seed=run_cfg.random_state,
        n_jobs=args.n_jobs,
    )


def main():
    args = parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
