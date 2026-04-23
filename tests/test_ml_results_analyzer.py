import importlib.util
import unittest
from pathlib import Path
import shutil

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "ml_results_analyzer.py"
SPEC = importlib.util.spec_from_file_location("ml_results_analyzer", MODULE_PATH)
ml_results_analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(ml_results_analyzer)


class TestMLResultsAnalyzer(unittest.TestCase):
    def setUp(self):
        raw_df = pd.DataFrame(
            [
                {
                    "run_id": "run_1",
                    "experiment_key": "sp1",
                    "test_dataset": "^GSPC_1d",
                    "symbol": "^GSPC",
                    "interval": "1d",
                    "model_name": "logistic",
                    "test_roc_auc": 0.50,
                    "test_accuracy": 0.50,
                    "total_return_pct": 3.0,
                    "buy_hold_return_pct": 10.0,
                    "excess_return_pct": -7.0,
                    "sharpe": 0.10,
                    "max_drawdown_pct": 20.0,
                    "signal_coverage": 0.50,
                    "num_trades": 100,
                },
                {
                    "run_id": "run_1",
                    "experiment_key": "a1",
                    "test_dataset": "AAPL_1d",
                    "symbol": "AAPL",
                    "interval": "1d",
                    "model_name": "logistic",
                    "test_roc_auc": 0.51,
                    "test_accuracy": 0.50,
                    "total_return_pct": -10.0,
                    "buy_hold_return_pct": 10.0,
                    "excess_return_pct": -20.0,
                    "sharpe": -0.20,
                    "max_drawdown_pct": 30.0,
                    "signal_coverage": 0.80,
                    "num_trades": 150,
                },
                {
                    "run_id": "run_1",
                    "experiment_key": "a2",
                    "test_dataset": "AAPL_1d",
                    "symbol": "AAPL",
                    "interval": "1d",
                    "model_name": "random_forest",
                    "test_roc_auc": 0.54,
                    "test_accuracy": 0.53,
                    "total_return_pct": 12.0,
                    "buy_hold_return_pct": 7.0,
                    "excess_return_pct": 5.0,
                    "sharpe": 0.40,
                    "max_drawdown_pct": 18.0,
                    "signal_coverage": 0.30,
                    "num_trades": 90,
                },
                {
                    "run_id": "run_2",
                    "experiment_key": "sp2",
                    "test_dataset": "^GSPC_1d",
                    "symbol": "^GSPC",
                    "interval": "1d",
                    "model_name": "logistic",
                    "test_roc_auc": 0.50,
                    "test_accuracy": 0.50,
                    "total_return_pct": 4.0,
                    "buy_hold_return_pct": 12.0,
                    "excess_return_pct": -8.0,
                    "sharpe": 0.10,
                    "max_drawdown_pct": 20.0,
                    "signal_coverage": 0.50,
                    "num_trades": 100,
                },
                {
                    "run_id": "run_2",
                    "experiment_key": "b1",
                    "test_dataset": "MSFT_1d",
                    "symbol": "MSFT",
                    "interval": "1d",
                    "model_name": "logistic",
                    "test_roc_auc": 0.58,
                    "test_accuracy": 0.56,
                    "total_return_pct": 8.0,
                    "buy_hold_return_pct": 4.0,
                    "excess_return_pct": 4.0,
                    "sharpe": 0.35,
                    "max_drawdown_pct": 14.0,
                    "signal_coverage": 0.75,
                    "num_trades": 120,
                },
                {
                    "run_id": "run_2",
                    "experiment_key": "b2",
                    "test_dataset": "MSFT_1d",
                    "symbol": "MSFT",
                    "interval": "1d",
                    "model_name": "random_forest",
                    "test_roc_auc": 0.55,
                    "test_accuracy": 0.54,
                    "total_return_pct": 6.0,
                    "buy_hold_return_pct": 5.0,
                    "excess_return_pct": 1.0,
                    "sharpe": 0.25,
                    "max_drawdown_pct": 11.0,
                    "signal_coverage": 0.25,
                    "num_trades": 70,
                },
            ]
        )
        self.df = ml_results_analyzer.add_sp500_outperformance(raw_df)

    def test_apply_filters_by_model_and_symbol(self):
        filtered = ml_results_analyzer.apply_filters(
            self.df,
            model_names=["logistic"],
            symbols=["MSFT"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["test_dataset"], "MSFT_1d")

    def test_build_model_summary_orders_by_mean_excess_return(self):
        summary = ml_results_analyzer.build_model_summary(self.df)
        self.assertEqual(summary.iloc[0]["model_name"], "random_forest")
        self.assertAlmostEqual(float(summary.iloc[0]["mean_excess_return_pct"]), 3.0)

    def test_build_dataset_winners_selects_best_row_per_dataset(self):
        winners = ml_results_analyzer.build_dataset_winners(
            self.df,
            sort_by="excess_return_pct",
            ascending=False,
        )
        winner_map = dict(zip(winners["test_dataset"], winners["model_name"]))
        self.assertEqual(winner_map["AAPL_1d"], "random_forest")
        self.assertEqual(winner_map["MSFT_1d"], "logistic")

    def test_add_sp500_outperformance_uses_interval_baseline(self):
        df = pd.DataFrame(
            [
                {
                    "run_id": "run_1",
                    "interval": "1d",
                    "symbol": "^GSPC",
                    "buy_hold_return_pct": 10.0,
                    "total_return_pct": 8.0,
                },
                {
                    "run_id": "run_1",
                    "interval": "1d",
                    "symbol": "AAPL",
                    "buy_hold_return_pct": 25.0,
                    "total_return_pct": 15.0,
                },
            ]
        )
        out = ml_results_analyzer.add_sp500_outperformance(df)
        aapl_row = out[out["symbol"] == "AAPL"].iloc[0]
        self.assertAlmostEqual(float(aapl_row["sp500_buy_hold_return_pct"]), 10.0)
        self.assertAlmostEqual(float(aapl_row["sp500_excess_return_pct"]), 5.0)

    def test_build_candle_diagnostics_computes_core_stats(self):
        candle_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=4, freq="D"),
                "position": [0, 1, 1, -1],
                "net_ret": [0.0, 0.01, -0.02, 0.03],
                "equity_curve": [1.0, 1.01, 0.9898, 1.019494],
            }
        )
        path = Path(__file__).resolve().parents[1] / "analysis" / "test_candle_diagnostics.csv"
        try:
            candle_df.to_csv(path, index=False)
            diag = ml_results_analyzer.build_candle_diagnostics(path)
        finally:
            if path.exists():
                path.unlink()

        self.assertEqual(diag["detail_rows"], 4)
        self.assertEqual(diag["detail_trade_days"], 3)
        self.assertEqual(diag["detail_turnover_events"], 2)
        self.assertAlmostEqual(diag["detail_final_equity"], 1.019494)

    def test_generate_plots_includes_risk_and_trade_regressions(self):
        tmp_dir = Path(__file__).resolve().parents[1] / "analysis" / "test_plot_output"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_df = self.df.copy()
            for idx, row in plot_df.iterrows():
                candle_df = pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2025-01-01", periods=4, freq="D"),
                        "next_return": [0.01, -0.02, 0.03, 0.01],
                        "equity_curve": [1.0, 1.01, 0.9898, 1.019494],
                    }
                )
                candle_path = tmp_dir / f"{row['test_dataset']}_{row['model_name']}.csv"
                candle_df.to_csv(candle_path, index=False)
                plot_df.at[idx, "candle_path"] = str(candle_path)

            model_summary = ml_results_analyzer.build_model_summary(plot_df)
            dataset_summary = ml_results_analyzer.build_dataset_summary(plot_df)
            correlation = ml_results_analyzer.build_metric_correlation(plot_df)

            stale_plot = tmp_dir / "stale_plot.png"
            stale_plot.write_text("old plot", encoding="utf-8")
            keep_file = tmp_dir / "notes.txt"
            keep_file.write_text("keep me", encoding="utf-8")

            plot_paths = ml_results_analyzer.generate_plots(
                df=plot_df,
                model_summary=model_summary,
                dataset_summary=dataset_summary,
                correlation=correlation,
                detailed_diagnostics=pd.DataFrame(),
                plots_dir=tmp_dir,
            )
            plot_names = {path.name for path in plot_paths}
            self.assertIn("risk_relationships.png", plot_names)
            self.assertIn("trade_return_regressions.png", plot_names)
            self.assertIn("best_outperformance_strategies.png", plot_names)
            self.assertIn("single_best_strategy_vs_benchmarks.png", plot_names)
            self.assertTrue((tmp_dir / "risk_relationships.png").exists())
            self.assertTrue((tmp_dir / "trade_return_regressions.png").exists())
            self.assertTrue((tmp_dir / "best_outperformance_strategies.png").exists())
            self.assertTrue((tmp_dir / "single_best_strategy_vs_benchmarks.png").exists())
            self.assertFalse(stale_plot.exists())
            self.assertTrue(keep_file.exists())
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
