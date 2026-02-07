import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "database" / "results.parquet"


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_parquet(path)


def expand_parameters(df: pd.DataFrame) -> pd.DataFrame:
    params = df["parameters"].apply(json.loads)
    params_df = pd.json_normalize(params)
    params_df.columns = [f"param_{c}" for c in params_df.columns]
    return pd.concat([df, params_df], axis=1)


def build_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg_map = {
        "return": "mean",
        "sharpe": "mean",
        "trades": "mean",
        "win_rate": "mean",
    }
    if "excess_return" in df.columns:
        agg_map["excess_return"] = "mean"

    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(agg_map)
        .rename(
            columns={
                "return": "return_mean",
                "sharpe": "sharpe_mean",
                "trades": "trades_mean",
                "win_rate": "win_rate_mean",
                "excess_return": "excess_return_mean",
            }
        )
    )

    # number of samples per group
    counts = df.groupby(group_cols, as_index=False).size().rename(columns={"size": "n"})
    return summary.merge(counts, on=group_cols, how="left")


def select_metric(summary: pd.DataFrame, metric: str | None) -> str:
    if metric:
        if metric not in summary.columns:
            raise ValueError(f"Metric '{metric}' not found in summary columns: {summary.columns.tolist()}")
        return metric

    if "excess_return_mean" in summary.columns:
        return "excess_return_mean"
    return "return_mean"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find underperforming individual parameter values (no parameter combos)."
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=3,
        help="Minimum number of samples required per parameter value.",
    )
    parser.add_argument(
        "--min-return-mean",
        type=float,
        default=None,
        help="Absolute threshold for return_mean (underperforming if below). Omit to disable.",
    )
    parser.add_argument(
        "--min-excess-return-mean",
        type=float,
        default=None,
        help="Absolute threshold for excess_return_mean (underperforming if below). Omit to disable.",
    )
    parser.add_argument(
        "--min-negative-rate",
        type=float,
        default=0.55,
        help="Minimum fraction of negative outcomes required to flag (e.g., 0.6 = 60%).",
    )
    parser.add_argument(
        "--min-return-diff",
        type=float,
        default=0.0,
        help="How much worse return_mean must be vs strategy average (e.g., 2.0 = at least 2% worse).",
    )
    parser.add_argument(
        "--min-excess-return-diff",
        type=float,
        default=0.0,
        help="How much worse excess_return_mean must be vs strategy average.",
    )
    parser.add_argument(
        "--no-median-check",
        action="store_true",
        help="Disable median-below-average requirement (more aggressive).",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all parameter values (otherwise only underperformers are saved).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/underperforming_param_values.csv",
        help="Where to save the underperforming parameter values (CSV).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric column to sort by (e.g., return_mean, excess_return_mean).",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Optional extra grouping columns (e.g., symbol interval).",
    )
    args = parser.parse_args()

    df = load_results(RESULTS_PATH)
    df = expand_parameters(df)

    param_cols = [c for c in df.columns if c.startswith("param_")]
    if not param_cols:
        raise ValueError("No parameter columns found to analyze.")

    long_params = df.melt(
        id_vars=[c for c in df.columns if not c.startswith("param_")],
        value_vars=param_cols,
        var_name="param_name",
        value_name="param_value",
    )

    group_key = ["strategy", "param_name", "param_value"] + args.group_by
    summary = build_summary(long_params, group_key)

    # Add distribution stats
    extra_stats = long_params.groupby(group_key).agg(
        return_median=("return", "median"),
        return_std=("return", "std"),
        neg_return_rate=("return", lambda s: (s < 0).mean()),
    ).reset_index()

    if "excess_return" in long_params.columns:
        excess_stats = long_params.groupby(group_key).agg(
            excess_return_median=("excess_return", "median"),
            excess_return_std=("excess_return", "std"),
            neg_excess_return_rate=("excess_return", lambda s: (s < 0).mean()),
        ).reset_index()
        summary = summary.merge(excess_stats, on=group_key, how="left")
    else:
        summary["excess_return_mean"] = pd.NA
        summary["excess_return_median"] = pd.NA
        summary["excess_return_std"] = pd.NA
        summary["neg_excess_return_rate"] = pd.NA

    summary = summary.merge(extra_stats, on=group_key, how="left")

    metric = select_metric(summary, args.metric)

    # Strategy-level averages (per strategy + optional group_by)
    baseline_group = ["strategy"] + args.group_by
    baseline = long_params.groupby(baseline_group, as_index=False).agg(
        strategy_return_mean=("return", "mean"),
        strategy_return_median=("return", "median"),
    )
    if "excess_return" in long_params.columns:
        baseline = baseline.merge(
            long_params.groupby(baseline_group, as_index=False).agg(
                strategy_excess_return_mean=("excess_return", "mean"),
                strategy_excess_return_median=("excess_return", "median"),
            ),
            on=baseline_group,
            how="left",
        )
    else:
        baseline["strategy_excess_return_mean"] = pd.NA
        baseline["strategy_excess_return_median"] = pd.NA

    # Flag underperforming parameter values (moderate, relative to strategy averages)
    eligible = summary[summary["n"] >= args.min_n].copy()
    eligible = eligible.merge(baseline, on=baseline_group, how="left")

    eligible["return_diff_vs_strategy"] = (
        eligible["return_mean"] - eligible["strategy_return_mean"]
    )
    if "excess_return_mean" in eligible.columns and eligible["excess_return_mean"].notna().any():
        eligible["excess_return_diff_vs_strategy"] = (
            eligible["excess_return_mean"] - eligible["strategy_excess_return_mean"]
        )
    else:
        eligible["excess_return_diff_vs_strategy"] = pd.NA

    def below_threshold(series: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series([True] * len(series), index=series.index)
        return series < threshold

    if "excess_return_mean" in eligible.columns and eligible["excess_return_mean"].notna().any():
        flag_mask = (
            below_threshold(eligible["return_mean"], args.min_return_mean)
            & below_threshold(eligible["excess_return_mean"], args.min_excess_return_mean)
            & (eligible["return_diff_vs_strategy"] <= -abs(args.min_return_diff))
            & (eligible["excess_return_diff_vs_strategy"] <= -abs(args.min_excess_return_diff))
            & (eligible["neg_return_rate"] >= args.min_negative_rate)
            & (eligible["neg_excess_return_rate"] >= args.min_negative_rate)
        )
        if not args.no_median_check:
            flag_mask = flag_mask & (
                (eligible["return_median"] < eligible["strategy_return_median"])
                & (eligible["excess_return_median"] < eligible["strategy_excess_return_median"])
            )
    else:
        flag_mask = (
            below_threshold(eligible["return_mean"], args.min_return_mean)
            & (eligible["return_diff_vs_strategy"] <= -abs(args.min_return_diff))
            & (eligible["neg_return_rate"] >= args.min_negative_rate)
        )
        if not args.no_median_check:
            flag_mask = flag_mask & (eligible["return_median"] < eligible["strategy_return_median"])

    eligible["flag_underperforming"] = flag_mask
    eligible = eligible.sort_values(metric, ascending=True)

    output_df = eligible if args.include_all else eligible[eligible["flag_underperforming"]]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print("Underperforming individual parameter values (no combos):")
    print(output_df.to_string(index=False))
    if args.include_all:
        print(
            f"\nFlagged {eligible['flag_underperforming'].sum()} of {len(eligible)} parameter values "
            f"(min_n={args.min_n}, min_negative_rate={args.min_negative_rate})."
        )
    print(f"\nSaved underperforming parameter values to: {output_path}")


if __name__ == "__main__":
    main()
