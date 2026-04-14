from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass
class EvaluationResult:
    model_name: str
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    accuracy: float
    per_class_recall: dict[str, float]
    confusion_matrix_labels: list[str]
    confusion_matrix_values: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_multiclass(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> EvaluationResult:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_recall: dict[str, float] = {}
    for label in labels:
        per_class_recall[label] = float(report.get(label, {}).get("recall", 0.0))

    return EvaluationResult(
        model_name=model_name,
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_recall=per_class_recall,
        confusion_matrix_labels=labels,
        confusion_matrix_values=cm.astype(int).tolist(),
    )
