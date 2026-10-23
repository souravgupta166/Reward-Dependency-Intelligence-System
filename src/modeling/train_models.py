
"""Train and compare purchase prediction models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "reward_customer_segments.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "metrics"

TARGET = "purchase"

# Only features available before the purchase outcome is observed.
FEATURES = [
    "tenure_months",
    "prior_orders",
    "avg_order_value_eur",
    "engagement_score",
    "discount_sensitivity",
    "reward_offered",
    "reward_amount_eur",
]


def evaluate_model(model, X_test, y_test):
    """Return held-out classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run the data generator and segmentation pipeline first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE)

    required = set(FEATURES + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data[FEATURES].isna().any().any():
        raise ValueError("Predictor columns contain missing values.")

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError("Purchase target must contain both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    results = []
    fitted_models = {}

    print("=" * 60)
    print("PURCHASE PREDICTION MODEL EVALUATION")
    print("=" * 60)
    print(f"Training customers: {len(X_train):,}")
    print(f"Test customers: {len(X_test):,}")
    print(f"Overall purchase rate: {y.mean():.1%}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results.append({"model": name, **metrics})
        fitted_models[name] = model

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "purchase_model_comparison.csv",
        index=False,
    )

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "target": TARGET,
            "model_name": best_name,
        },
        MODEL_DIR / "purchase_model.joblib",
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False, float_format="%.4f"))
    print(f"\nBest model by test ROC-AUC: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'purchase_model.joblib'}")
    print(
        "Saved metrics: "
        f"{REPORT_DIR / 'purchase_model_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
