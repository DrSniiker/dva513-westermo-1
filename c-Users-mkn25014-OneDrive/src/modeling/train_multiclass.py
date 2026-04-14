from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from evaluate_multiclass import evaluate_multiclass
from features import build_feature_bundle
from splits import stratified_split


BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_CSV = BASE_DIR / "archive" / "UNSW_NB15_training-set.csv"
TEST_CSV = BASE_DIR / "archive" / "UNSW_NB15_testing-set.csv"
TARGET_COL = "attack_cat"
RANDOM_STATE = 42

OUTPUT_DIR = BASE_DIR / "reports"
METRICS_JSON = OUTPUT_DIR / "unsw_multiclass_metrics.json"
SUMMARY_MD = OUTPUT_DIR / "unsw_multiclass_eval.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-official-test", action="store_true")
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "UNSW files not found. Expected:\n"
            f"- {TRAIN_CSV}\n"
            f"- {TEST_CSV}"
        )
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)


def fit_predict_model(
    model_name: str,
    estimator,
    x_train_df: pd.DataFrame,
    y_train: np.ndarray,
    x_eval_df: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    bundle = build_feature_bundle(x_train_df)
    steps = [("preprocessor", bundle.preprocessor)]
    if model_name == "hist_gradient_boosting":
        steps.append(("to_dense", FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x)))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps=steps)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(x_train_df[bundle.feature_columns], y_train, model__sample_weight=sample_weight)
    y_pred = pipeline.predict(x_eval_df[bundle.feature_columns])
    return y_pred


def save_report(results: list[dict], labels: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps({"labels": labels, "models": results}, indent=2), encoding="utf-8")

    lines = [
        "# UNSW Multiclass Evaluation",
        "",
        "| Model | Macro F1 | Weighted F1 | Balanced Acc | Acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model_name']} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
            f"{r['balanced_accuracy']:.4f} | {r['accuracy']:.4f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_df, test_df = load_data()

    if args.use_official_test:
        fit_df = train_df.copy()
        eval_df = test_df.copy()
    else:
        fit_df, eval_df = stratified_split(train_df, target_col=TARGET_COL, random_state=RANDOM_STATE)

    fit_df = fit_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    eval_df = eval_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    labels = sorted(fit_df[TARGET_COL].astype(str).unique().tolist())
    y_fit = fit_df[TARGET_COL].astype(str).to_numpy()
    y_eval = eval_df[TARGET_COL].astype(str).to_numpy()

    models = {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results: list[dict] = []
    for model_name, estimator in models.items():
        y_pred = fit_predict_model(
            model_name=model_name,
            estimator=estimator,
            x_train_df=fit_df,
            y_train=y_fit,
            x_eval_df=eval_df,
            labels=labels,
        )
        eval_result = evaluate_multiclass(model_name, y_eval, y_pred, labels)
        result_dict = eval_result.to_dict()
        results.append(result_dict)
        print(
            f"{model_name}: macro_f1={result_dict['macro_f1']:.4f}, "
            f"weighted_f1={result_dict['weighted_f1']:.4f}, "
            f"balanced_acc={result_dict['balanced_accuracy']:.4f}, acc={result_dict['accuracy']:.4f}"
        )

    save_report(results, labels)
    print(f"Saved report files:\n- {METRICS_JSON}\n- {SUMMARY_MD}")


if __name__ == "__main__":
    main()
