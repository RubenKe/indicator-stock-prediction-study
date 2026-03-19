from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED_MODELS = ("logistic", "random_forest", "gradient_boosting")
SUPPORTED_PROFILES = ("standard",)


def parse_model_list(models_csv: str) -> list[str]:
    parsed = [m.strip() for m in models_csv.split(",") if m.strip()]
    parsed = list(dict.fromkeys(parsed))
    if not parsed:
        raise ValueError("At least one model must be provided.")

    unknown = [m for m in parsed if m not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(
            f"Unsupported model(s): {unknown}. Supported: {list(SUPPORTED_MODELS)}"
        )
    return parsed


def validate_profile(profile: str) -> str:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported profile '{profile}'. Supported: {list(SUPPORTED_PROFILES)}"
        )
    return profile


def build_estimator(model_name: str, random_state: int, n_jobs: int):
    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        penalty="l2",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)

    raise KeyError(f"Unknown model_name: {model_name}")


def get_param_grid(model_name: str, profile: str) -> dict:
    validate_profile(profile)
    if model_name == "logistic":
        return {"clf__C": [0.01, 0.1, 1, 10]}

    if model_name == "random_forest":
        return {
            "n_estimators": [200, 500],
            "max_depth": [3, 6, 10],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt"],
        }

    if model_name == "gradient_boosting":
        return {
            "n_estimators": [100, 300, 700],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 4],
            "subsample": [0.7, 1.0],
        }

    raise KeyError(f"Unknown model_name: {model_name}")
