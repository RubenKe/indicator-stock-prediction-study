import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLUMNS, build_feature_frame, engineer_features
from ml.persistence import save_model_artifacts
from ml.train_eval import build_group_kfold_splits, build_signal_frame


class TestMLPipeline(unittest.TestCase):
    def test_feature_engineering_uses_shifted_high_for_breakout(self):
        n = 1400
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = np.full(n, 11.0)
        high = np.full(n, 12.0)
        low = np.full(n, 10.0)
        open_ = np.full(n, 11.0)
        volume = np.full(n, 1000.0)

        # Inject a large current-bar high. With shifted breakout logic this row should
        # still compare against the previous window highs (12), not this row's 100.
        spike_idx = 500
        high[spike_idx] = 100.0

        raw = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        engineered = engineer_features(raw)

        expected = (11.0 - 12.0) / 12.0
        self.assertAlmostEqual(
            float(engineered.iloc[spike_idx]["breakout_dist_20"]),
            expected,
            places=10,
        )

        final_frame = build_feature_frame(
            raw_df=raw,
            dataset_id="TEST_15m",
            symbol="TEST",
            interval="15m",
            test_candles=1250,
        )
        self.assertEqual(len(final_frame), 1250)
        self.assertFalse(final_frame[FEATURE_COLUMNS + ["next_return", "y"]].isna().any().any())

    def test_group_kfold_splits_are_group_disjoint(self):
        groups = np.array(["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8)
        splits = build_group_kfold_splits(groups=groups, n_splits=4)
        self.assertEqual(len(splits), 4)

        for train_idx, val_idx in splits:
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_signal_math_matches_expected_values(self):
        p_up = np.array([0.60, 0.40, 0.50, 0.70], dtype=float)
        next_returns = pd.Series([0.01, -0.02, 0.03, -0.01], dtype=float)
        signal_df = build_signal_frame(
            probabilities=p_up,
            next_returns=next_returns,
            threshold_long=0.55,
            threshold_short=0.45,
            commission=0.001,
            index=pd.RangeIndex(0, 4),
        )

        self.assertListEqual(signal_df["signal_raw"].astype(int).tolist(), [1, -1, 0, 1])
        self.assertListEqual(signal_df["position"].astype(int).tolist(), [0, 1, -1, 0])

        expected_cost = [0.0, 0.001, 0.002, 0.001]
        for got, exp in zip(signal_df["cost"].tolist(), expected_cost):
            self.assertAlmostEqual(got, exp, places=12)

        expected_net = [0.0, -0.021, -0.032, -0.001]
        for got, exp in zip(signal_df["net_ret"].tolist(), expected_net):
            self.assertAlmostEqual(got, exp, places=12)

    def test_model_persistence_round_trip(self):
        X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y = np.array([0, 0, 1, 1], dtype=int)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
            ]
        )
        model.fit(X, y)
        expected_probs = model.predict_proba(X)[:, 1]

        candle_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
                "symbol": ["TEST"] * 4,
                "interval": ["15m"] * 4,
                "p_up": expected_probs,
                "signal_raw": [0, 0, 0, 0],
                "position": [0, 0, 0, 0],
                "next_return": [0.0, 0.0, 0.0, 0.0],
                "gross_ret": [0.0, 0.0, 0.0, 0.0],
                "cost": [0.0, 0.0, 0.0, 0.0],
                "net_ret": [0.0, 0.0, 0.0, 0.0],
                "equity_curve": [1.0, 1.0, 1.0, 1.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = save_model_artifacts(
                run_id="run_test",
                test_dataset="TEST_15m",
                model_name="logistic",
                estimator=model,
                candle_df=candle_df,
                metadata={"key": "value"},
                model_registry_dir=tmp_path / "registry",
                analysis_ml_dir=tmp_path / "analysis",
            )
            reloaded = joblib.load(paths["model_path"])
            got_probs = reloaded.predict_proba(X)[:, 1]
            self.assertTrue(np.allclose(got_probs, expected_probs))


if __name__ == "__main__":
    unittest.main()
