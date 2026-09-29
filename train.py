"""
train.py — Train the stroke risk prediction pipeline with MLflow experiment tracking.

Logs 4 runs under the "stroke-risk-prediction" experiment: the 3 base learners
(Logistic Regression, Random Forest, HistGradientBoosting) evaluated individually,
plus the final Stacking Ensemble — so you can compare them side-by-side in the
MLflow UI instead of only reading a static results table.

Usage:
    python train.py
    python train.py --data data/healthcare-dataset-stroke-data.csv
    python train.py --model-out model/stroke_pipeline.pkl

View results after running:
    mlflow ui
    # then open http://127.0.0.1:5000 in a browser
"""

import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.utils import resample

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
EXPERIMENT_NAME = "stroke-risk-prediction"


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-motivated features before preprocessing."""
    df = df.copy()

    # Clinical glucose tiers (ADA definitions)
    df["glucose_risk"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 100, 125, 999],
        labels=["Normal", "Prediabetic", "Diabetic"]
    ).astype(str)

    # BMI category (WHO)
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 999],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    ).astype(str)

    # Age risk tier — stroke risk accelerates sharply after 65
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 65, 999],
        labels=["Under40", "40to55", "55to65", "Over65"]
    ).astype(str)

    # Comorbidity score: number of known cardiovascular risk factors
    df["comorbidity_score"] = df["hypertension"].astype(int) + df["heart_disease"].astype(int)

    # Interaction: age × hypertension (older hypertensive patients = highest risk)
    df["age_x_hypertension"] = df["age"] * df["hypertension"]

    # Interaction: age × heart_disease
    df["age_x_heart_disease"] = df["age"] * df["heart_disease"]

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_preprocessor():
    numeric_features = [
        "age", "avg_glucose_level", "bmi",
        "comorbidity_score", "age_x_hypertension", "age_x_heart_disease"
    ]
    binary_features = ["hypertension", "heart_disease"]
    categorical_features = [
        "gender", "ever_married", "work_type", "Residence_type",
        "smoking_status", "glucose_risk", "bmi_category", "age_group"
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("bin", binary_transformer, binary_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor


# ── Candidate models ──────────────────────────────────────────────────────────
# Each entry: (friendly run name, model instance, params dict to log to MLflow)

def build_candidates():
    candidates = {}

    lr_params = dict(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    candidates["logistic_regression"] = (
        LogisticRegression(**lr_params), {"model_type": "LogisticRegression", **lr_params}
    )

    rf_params = dict(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    candidates["random_forest"] = (
        RandomForestClassifier(**rf_params), {"model_type": "RandomForestClassifier", **rf_params}
    )

    hgb_params = dict(
        max_iter=200, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    candidates["hist_gradient_boosting"] = (
        HistGradientBoostingClassifier(**hgb_params),
        {"model_type": "HistGradientBoostingClassifier", **hgb_params}
    )

    # Stacking ensemble reuses the same base estimators + a fresh LR meta-learner
    base_estimators = [
        ("lr", LogisticRegression(**lr_params)),
        ("rf", RandomForestClassifier(**rf_params)),
        ("hgb", HistGradientBoostingClassifier(**hgb_params)),
    ]
    meta_learner = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        passthrough=False,
        n_jobs=-1
    )
    candidates["stacking_ensemble"] = (
        stacking_model,
        {
            "model_type": "StackingClassifier",
            "base_estimators": "lr+rf+hgb",
            "meta_learner": "LogisticRegression",
            "cv_folds": 5,
        }
    )
    return candidates


# ── Training helpers ──────────────────────────────────────────────────────────

def load_and_prepare(data_path):
    df = pd.read_csv(data_path)
    df = df[df["gender"] != "Other"].reset_index(drop=True)  # 1 row
    df = engineer_features(df)
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]
    return X, y


def oversample_training(X_proc, y):
    """Resample minority class to 1:4 ratio on processed training data."""
    y = np.array(y)
    X_maj = X_proc[y == 0]
    X_min = X_proc[y == 1]
    target = len(X_maj) // 4
    X_min_over = resample(X_min, n_samples=target, replace=True, random_state=RANDOM_STATE)
    y_min_over = np.ones(target, dtype=int)
    X_bal = np.vstack([X_maj, X_min_over])
    y_bal = np.concatenate([np.zeros(len(X_maj), dtype=int), y_min_over])
    idx = np.random.RandomState(RANDOM_STATE).permutation(len(X_bal))
    return X_bal[idx], y_bal[idx]


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
    }


def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 200)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1s)]), float(max(f1s))


def run_and_log(run_name, model, params, preprocessor,
                 X_train_bal, y_train_bal, X_test_proc, y_test):
    """Fit one candidate model, log params/metrics/model to MLflow, return metrics."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", params.get("model_type", run_name))
        mlflow.log_params(params)
        mlflow.log_param("oversample_ratio", "1:4")
        mlflow.log_param("n_train_samples_balanced", len(y_train_bal))

        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        best_thresh, best_f1 = find_best_threshold(np.array(y_test), y_proba)
        y_pred_opt = (y_proba >= best_thresh).astype(int)
        metrics_opt = evaluate(y_test, y_pred_opt, y_proba)
        mlflow.log_metric("optimal_threshold", best_thresh)
        for k, v in metrics_opt.items():
            mlflow.log_metric(f"opt_{k}", v)

        # Log a deployable pipeline (preprocessor + fitted model) as one MLflow model,
        # so it can be reloaded and used for inference without re-fitting anything.
        inference_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        mlflow.sklearn.log_model(
            inference_pipeline,
            artifact_path="model",
            registered_model_name=None,  # set to a string to auto-register in Model Registry
            serialization_format="cloudpickle",  # MLflow 3.x defaults to skops, which rejects tree models
        )

        print(f"  [{run_name}] default: {metrics}  |  opt@{best_thresh:.3f}: {metrics_opt}")
        return metrics, metrics_opt, best_thresh


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/healthcare-dataset-stroke-data.csv")
    parser.add_argument("--model-out", default="model/stroke_pipeline.pkl")
    args = parser.parse_args()

    print("\n" + "─" * 60)
    print("  Stroke Risk Prediction — Training Pipeline (MLflow-tracked)")
    print("─" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load
    print("\nLoading data...")
    X, y = load_and_prepare(args.data)
    print(f"  Shape: {X.shape} | Positive rate: {y.mean():.1%}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocess
    print("\nPreprocessing...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print(f"  Feature dimensions: {X_train_proc.shape[1]}")

    # Oversample
    print("\nOversampling training set...")
    X_train_bal, y_train_bal = oversample_training(X_train_proc, y_train)
    print(f"  Balanced shape: {X_train_bal.shape} | Positive rate: {y_train_bal.mean():.1%}")

    # Train + log every candidate under one experiment
    print(f"\nTraining candidates and logging to MLflow experiment '{EXPERIMENT_NAME}'...")
    candidates = build_candidates()
    results = {}
    for run_name, (model, params) in candidates.items():
        metrics, metrics_opt, best_thresh = run_and_log(
            run_name, model, params, preprocessor,
            X_train_bal, y_train_bal, X_test_proc, y_test
        )
        results[run_name] = {
            "model": model, "metrics": metrics,
            "metrics_opt": metrics_opt, "threshold": best_thresh
        }

    # Print a comparison table (mirrors what you'll see in the MLflow UI)
    print("\n" + "─" * 60)
    print("  Comparison — Default Threshold (0.5)")
    print("─" * 60)
    print(f"  {'Model':<26}{'Acc':<8}{'Prec':<8}{'Recall':<8}{'F1':<8}{'ROC-AUC':<8}")
    for name, r in results.items():
        m = r["metrics"]
        print(f"  {name:<26}{m['accuracy']:<8}{m['precision']:<8}{m['recall']:<8}{m['f1']:<8}{m['roc_auc']:<8}")

    # Save the stacking ensemble as the production pickle bundle (unchanged behavior
    # for the existing Streamlit app / app.py, which expects this exact structure)
    best = results["stacking_ensemble"]
    pipeline_bundle = {
        "preprocessor": preprocessor,
        "model": best["model"],
        "threshold": best["threshold"],
        "feature_names": list(X.columns),
    }
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(pipeline_bundle, f)
    print(f"\n  Production model (stacking ensemble) saved to: {args.model_out}")
    print(f"  Run `mlflow ui` and open http://127.0.0.1:5000 to compare all 4 runs.")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
