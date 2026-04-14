import json
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Dataset paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "IEC-61850" / "preprocessed csv files" / "preprocessed csv files"
DATASETS = {
    "message_injection": {
        "train": DATA_DIR / "message_injection_train.csv",
        "test": DATA_DIR / "message_injection_test.csv",
    },
    "masquerade": {
        "train": DATA_DIR / "masquerade_train.csv",
        "test": DATA_DIR / "masquerade_test.csv",
    },
    "replay": {
        "train": DATA_DIR / "replay_train.csv",
        "test": DATA_DIR / "replay_test.csv",
    },
    "poisoning": {
        "train": DATA_DIR / "poisoning_train.csv",
        "test": DATA_DIR / "poisoning_test.csv",
    },
}

LABEL_COL = "target"
DROP_COLS = ["frame.number", "id", "attack_cat"]
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_JSON = OUTPUT_DIR / "goose_baseline_results.json"

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
SEED = 42
VALIDATION_SPLIT = 0.2
USE_CLASS_IMBALANCE_HANDLING = True
USE_THRESHOLD_TUNING = True
THRESHOLD_CANDIDATES = np.arange(0.10, 0.91, 0.05)
SCENARIO_THRESHOLD_FALLBACKS = {
    "default": 0.5,
    "masquerade": 0.9,
}
PRUNING_RATIOS = [0.20, 0.40, 0.60, 0.80, 0.95]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def dataset_sanity_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_label = train_df[LABEL_COL].value_counts(dropna=False).to_dict()
    test_label = test_df[LABEL_COL].value_counts(dropna=False).to_dict()

    feature_df = train_df.drop(columns=[LABEL_COL], errors="ignore")
    categorical_count = int(feature_df.select_dtypes(include=["object", "category"]).shape[1])
    numeric_count = int(feature_df.shape[1] - categorical_count)

    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_class_distribution": {str(k): int(v) for k, v in train_label.items()},
        "test_class_distribution": {str(k): int(v) for k, v in test_label.items()},
        "train_missing_values_total": int(train_df.isna().sum().sum()),
        "test_missing_values_total": int(test_df.isna().sum().sum()),
        "feature_type_counts": {
            "categorical": categorical_count,
            "numeric_or_bool": numeric_count,
        },
        "contains_goose_columns": bool(any(col.startswith("goose.") for col in feature_df.columns)),
    }


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    train = train_df.copy()
    test = test_df.copy()

    drop_actual = [col for col in drop_cols if col in train.columns or col in test.columns]
    if drop_actual:
        train = train.drop(columns=drop_actual, errors="ignore")
        test = test.drop(columns=drop_actual, errors="ignore")

    train = train.dropna(subset=[LABEL_COL]).copy()
    test = test.dropna(subset=[LABEL_COL]).copy()
    train[LABEL_COL] = train[LABEL_COL].astype(int)
    test[LABEL_COL] = test[LABEL_COL].astype(int)

    x_train_raw = train.drop(columns=[LABEL_COL])
    y_train = train[LABEL_COL].to_numpy(dtype=np.float32)
    x_test_raw = test.drop(columns=[LABEL_COL])
    y_test = test[LABEL_COL].to_numpy(dtype=np.float32)
    return x_train_raw, y_train, x_test_raw, y_test, sorted(drop_actual)


def preprocess(
    x_train_raw: pd.DataFrame,
    x_val_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat_cols = x_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train_raw.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    x_train = preprocessor.fit_transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)
    x_test = preprocessor.transform(x_test_raw)

    if hasattr(x_train, "toarray"):
        x_train = x_train.toarray()
        x_val = x_val.toarray()
        x_test = x_test.toarray()

    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


class SimpleANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


def train_model(
    model: nn.Module,
    x_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    pos_weight: float,
) -> list[float]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(x_train_t.size(0))
        epoch_loss = 0.0

        for i in range(0, x_train_t.size(0), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            xb = x_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, (x_train_t.size(0) // BATCH_SIZE))
        losses.append(float(avg_loss))
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - Loss: {avg_loss:.4f}")
    return losses


def evaluate_with_threshold(
    model: nn.Module,
    x_eval_t: torch.Tensor,
    y_eval: np.ndarray,
    threshold: float,
) -> dict:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_eval_t)).squeeze().cpu().numpy()
    y_pred = (probs >= threshold).astype(int)
    y_true = y_eval.astype(int)
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def tune_threshold(model: nn.Module, x_val_t: torch.Tensor, y_val: np.ndarray) -> tuple[float, dict]:
    best_threshold = SCENARIO_THRESHOLD_FALLBACKS["default"]
    best_metrics = evaluate_with_threshold(model, x_val_t, y_val, threshold=best_threshold)
    best_score = best_metrics["mcc"] + (0.25 * best_metrics["f1"])

    for threshold in THRESHOLD_CANDIDATES:
        metrics = evaluate_with_threshold(model, x_val_t, y_val, threshold=float(threshold))
        score = metrics["mcc"] + (0.25 * metrics["f1"])
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def model_size_stats(model: nn.Module) -> dict:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_fp32 = int(params * 4)
    return {
        "total_params": int(params),
        "trainable_params": int(trainable),
        "estimated_size_mb_fp32": float(bytes_fp32 / (1024 ** 2)),
    }


def sparsity_stats(model: nn.Module) -> dict:
    total = 0
    zeros = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach()
        total += int(weight.numel())
        zeros += int((weight == 0).sum().item())
        if module.bias is not None:
            bias = module.bias.detach()
            total += int(bias.numel())
            zeros += int((bias == 0).sum().item())
    ratio = float(zeros / max(1, total))
    return {"total_params_checked": int(total), "zero_params": int(zeros), "sparsity_ratio": ratio}


def latency_benchmark(model: nn.Module, x_test_t: torch.Tensor, runs: int = 20) -> dict:
    model.eval()
    out = {}
    for batch_size in [1, 8, 32]:
        if x_test_t.size(0) < batch_size:
            continue
        samples = x_test_t[:batch_size]
        with torch.no_grad():
            _ = model(samples)
        start = perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(samples)
        elapsed = perf_counter() - start
        mean_ms = (elapsed / runs) * 1000.0
        out[f"batch_{batch_size}"] = {
            "mean_ms_per_batch": float(mean_ms),
            "mean_ms_per_sample": float(mean_ms / batch_size),
            "runs": int(runs),
        }
    return out


def run_pruning_sweep(
    model: nn.Module,
    x_test_t: torch.Tensor,
    y_test: np.ndarray,
    threshold: float,
) -> list[dict]:
    results = []
    for ratio in PRUNING_RATIOS:
        pruned_model = deepcopy(model)
        parameters_to_prune = []
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, "weight"))
        if not parameters_to_prune:
            continue

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")

        metrics = evaluate_with_threshold(pruned_model, x_test_t, y_test, threshold=threshold)
        size = model_size_stats(pruned_model)
        sparsity = sparsity_stats(pruned_model)
        latency = latency_benchmark(pruned_model, x_test_t)
        results.append(
            {
                "target_pruning_ratio": float(ratio),
                "decision_threshold": float(threshold),
                "metrics": metrics,
                "model_stats": size,
                "sparsity": sparsity,
                "latency": latency,
            }
        )
    return results


def run_experiment(
    scenario_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_csv: Path,
    test_csv: Path,
) -> dict:
    x_train_raw, y_train, x_test_raw, y_test, dropped = prepare_xy(train_df, test_df, DROP_COLS)

    x_train_fit_raw, x_val_raw, y_train_fit, y_val = train_test_split(
        x_train_raw,
        y_train,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=y_train.astype(int),
    )
    x_train, x_val, x_test = preprocess(x_train_fit_raw, x_val_raw, x_test_raw)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_fit, dtype=torch.float32).view(-1, 1)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    positive = int(np.sum(y_train_fit == 1))
    negative = int(np.sum(y_train_fit == 0))
    if USE_CLASS_IMBALANCE_HANDLING and positive > 0:
        pos_weight = float(negative / max(1, positive))
    else:
        pos_weight = 1.0

    model = SimpleANN(input_dim=x_train.shape[1])
    losses = train_model(model, x_train_t, y_train_t, pos_weight=pos_weight)

    threshold_source = "fallback"
    threshold = SCENARIO_THRESHOLD_FALLBACKS.get(scenario_name, SCENARIO_THRESHOLD_FALLBACKS["default"])
    val_metrics = evaluate_with_threshold(model, x_val_t, y_val, threshold=threshold)
    if USE_THRESHOLD_TUNING:
        tuned_threshold, tuned_val_metrics = tune_threshold(model, x_val_t, y_val)
        threshold = tuned_threshold
        val_metrics = tuned_val_metrics
        threshold_source = "validation_tuned"

    baseline_metrics = evaluate_with_threshold(model, x_test_t, y_test, threshold=threshold)
    pruning_results = run_pruning_sweep(model, x_test_t, y_test, threshold=threshold)

    model_stats = model_size_stats(model)
    sparsity = sparsity_stats(model)
    latency = latency_benchmark(model, x_test_t)
    batch_1 = latency.get("batch_1", {})

    print(
        f"[{scenario_name}] Baseline -> MCC: {baseline_metrics['mcc']:.4f} | "
        f"F1: {baseline_metrics['f1']:.4f} | ACC: {baseline_metrics['accuracy']:.4f} | "
        f"Threshold: {threshold:.2f} ({threshold_source})"
    )
    for pr in pruning_results:
        ratio_pct = int(pr["target_pruning_ratio"] * 100)
        m = pr["metrics"]
        print(
            f"[{scenario_name}] Prune {ratio_pct}% -> MCC: {m['mcc']:.4f} | "
            f"F1: {m['f1']:.4f} | ACC: {m['accuracy']:.4f}"
        )

    return {
        "scenario": scenario_name,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "dropped_columns": dropped,
        "input_dim": int(x_train.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "class_imbalance_handling": bool(USE_CLASS_IMBALANCE_HANDLING),
        "positive_class_weight": float(pos_weight),
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
        "validation_metrics": val_metrics,
        "last_epoch_loss": float(losses[-1]),
        "baseline_metrics": baseline_metrics,
        "inference_time_ms_per_sample": float(batch_1.get("mean_ms_per_sample", 0.0)),
        "model_stats": model_stats,
        "sparsity": sparsity,
        "latency": latency,
        "pruning_sweep": pruning_results,
    }


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sanity_reports: dict[str, dict] = {}
    experiments: list[dict] = []

    for name, split in DATASETS.items():
        print(f"\n=== Running scenario: {name} ===")
        train_csv = split["train"]
        test_csv = split["test"]
        train_df, test_df = read_split(train_csv, test_csv)
        sanity = dataset_sanity_report(train_df, test_df)
        sanity_reports[name] = sanity

        print(f"Sanity [{name}] train={sanity['train_rows']} test={sanity['test_rows']}")
        print(f"Sanity [{name}] class_dist_train={sanity['train_class_distribution']}")

        result = run_experiment(
            scenario_name=name,
            train_df=train_df,
            test_df=test_df,
            train_csv=train_csv,
            test_csv=test_csv,
        )
        experiments.append(result)

    print("\n=== Final scenario summary ===")
    for exp in experiments:
        m = exp["baseline_metrics"]
        print(
            f"{exp['scenario']}: baseline ACC={m['accuracy']:.4f}, "
            f"F1={m['f1']:.4f}, MCC={m['mcc']:.4f}, TH={exp['decision_threshold']:.2f}"
        )

    payload = {
        "seed": SEED,
        "dataset_family": "IEC-61850 GOOSE",
        "data_dir": str(DATA_DIR),
        "pruning_ratios": PRUNING_RATIOS,
        "threshold_candidates": [float(x) for x in THRESHOLD_CANDIDATES.tolist()],
        "sanity": sanity_reports,
        "experiments": experiments,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved baseline+pruning results to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
