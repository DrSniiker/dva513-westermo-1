import argparse
import json
import random
import warnings
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import murmurhash3_32

# Dataset paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "IEC-61850" / "preprocessed csv files" / "preprocessed csv files"
SCENARIO_FILES = {
    "message_injection": ("message_injection_train.csv", "message_injection_test.csv"),
    "masquerade": ("masquerade_train.csv", "masquerade_test.csv"),
    "replay": ("replay_train.csv", "replay_test.csv"),
    "poisoning": ("poisoning_train.csv", "poisoning_test.csv"),
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
EPOCH_LOG_INTERVAL = 5
MIN_ACCEPTABLE_TUNED_F1 = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTAMINATION_WARN_RATIO = 0.01
LATENCY_RUNS = 200
LATENCY_WARMUP = 20
QUIET = False
ENABLE_PRUNING = True
SAVE_MODEL = False


def parse_args() -> argparse.Namespace:
    # Runtime knobs are exposed via CLI to keep experiments reproducible.
    parser = argparse.ArgumentParser(description="IEC-61850 ANN baseline + pruning sweep")
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=["message_injection", "masquerade", "replay", "poisoning", "all"],
        default=["all"],
        help="Which scenario(s) to run (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to preprocessed CSV directory",
    )
    parser.add_argument("--output-json", type=str, default="results.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch logs")
    parser.add_argument("--no-pruning", action="store_true", help="Skip the pruning sweep")
    parser.add_argument("--save-model", action="store_true", help="Save model weights and preprocessor")
    return parser.parse_args()


def resolve_datasets(data_dir: Path, selected_scenarios: list[str]) -> dict[str, dict[str, Path]]:
    # Build a map like {"masquerade": {"train": "...", "test": "..."}}.
    return {
        name: {
            "train": data_dir / SCENARIO_FILES[name][0],
            "test": data_dir / SCENARIO_FILES[name][1],
        }
        for name in selected_scenarios
    }


def seed_everything(seed: int = 42) -> None:
    # Force deterministic behavior as much as possible across NumPy/PyTorch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_metrics(metrics: dict) -> str:
    return (
        f"ACC={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"MCC={metrics['mcc']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f}"
    )


def read_split(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def dataset_sanity_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    # Quick profile used for logging and debugging data issues before training.
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
    # Remove known leakage/meta columns, keep label, and split into features/target.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    # Infer column types from training fold only.
    cat_cols = x_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train_raw.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    pipe = Pipeline([("pre", preprocessor)])

    # Fit ONLY on training fold to avoid preprocessing leakage.
    x_train = pipe.fit_transform(x_train_raw)
    x_val = pipe.transform(x_val_raw)
    x_test = pipe.transform(x_test_raw)

    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32), pipe


class SimpleANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Small MLP: enough capacity for baseline performance and pruning experiments.
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
    pos_weight_t: torch.Tensor,
    scenario_name: str,
) -> list[float]:
    # BCEWithLogitsLoss combines sigmoid + BCE in a numerically stable way.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        # Shuffle every epoch for better stochastic training behavior.
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
        epoch_num = epoch + 1
        should_log = epoch_num == 1 or epoch_num == EPOCHS or epoch_num % EPOCH_LOG_INTERVAL == 0
        if should_log and not QUIET:
            print(f"  [{scenario_name}] Epoch {epoch_num:02d}/{EPOCHS} | loss={avg_loss:.4f}")
    return losses


def evaluate_with_threshold(
    model: nn.Module,
    x_eval_t: torch.Tensor,
    y_eval: np.ndarray,
    threshold: float,
) -> dict:
    # Convert logits -> probabilities -> hard labels with a configurable threshold.
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


def tune_threshold(model: nn.Module, x_val_t: torch.Tensor, y_val: np.ndarray) -> tuple[float, dict, str | None]:
    # Choose threshold that maximizes validation F1 over a predefined grid.
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_val_t)).squeeze().cpu().numpy()

    y_true = y_val.astype(int)
    best_threshold = float(SCENARIO_THRESHOLD_FALLBACKS["default"])
    best_metrics: dict | None = None
    best_f1 = -1.0

    for threshold in THRESHOLD_CANDIDATES:
        y_pred = (probs >= float(threshold)).astype(int)
        metrics = {
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        if metrics["f1"] > best_f1:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = metrics["f1"]

    if best_metrics is None:
        best_metrics = evaluate_with_threshold(model, x_val_t, y_val, threshold=best_threshold)
        best_f1 = best_metrics["f1"]

    warning_msg = None
    if best_f1 < MIN_ACCEPTABLE_TUNED_F1:
        warning_msg = (
            f"Threshold tuning found weak validation F1={best_f1:.4f}. "
            "Model may not have learned the positive class."
        )
        warnings.warn(warning_msg, RuntimeWarning)

    return best_threshold, best_metrics, warning_msg


def check_train_test_overlap(
    x_train: np.ndarray,
    x_test: np.ndarray,
    warn_ratio: float = CONTAMINATION_WARN_RATIO,
) -> dict:
    # Hash each transformed row to detect accidental train/test contamination.
    if x_test.shape[0] == 0:
        return {
            "train_unique_hashes": 0,
            "test_rows": 0,
            "overlap_rows": 0,
            "overlap_ratio": 0.0,
            "warning": None,
        }

    train_hashes = {murmurhash3_32(row.tobytes()) for row in x_train}
    test_hashes = [murmurhash3_32(row.tobytes()) for row in x_test]
    overlap_rows = int(sum(h in train_hashes for h in test_hashes))
    overlap_ratio = float(overlap_rows / len(test_hashes))

    warning_msg = None
    if overlap_ratio > warn_ratio:
        warning_msg = (
            f"{overlap_rows} test rows ({100.0 * overlap_ratio:.1f}%) appear in training set - "
            "possible data contamination."
        )
        warnings.warn(warning_msg, RuntimeWarning)

    return {
        "train_unique_hashes": int(len(train_hashes)),
        "test_rows": int(len(test_hashes)),
        "overlap_rows": overlap_rows,
        "overlap_ratio": overlap_ratio,
        "warning": warning_msg,
    }


def model_size_stats(model: nn.Module) -> dict:
    # Approximate footprint assuming fp32 parameters (4 bytes each).
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_fp32 = int(params * 4)
    return {
        "total_params": int(params),
        "trainable_params": int(trainable),
        "estimated_size_mb_fp32": float(bytes_fp32 / (1024 ** 2)),
    }


def sparsity_stats(model: nn.Module) -> dict:
    # Counts zeros after pruning to report effective sparsity.
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


def latency_benchmark(
    model: nn.Module,
    x_test_t: torch.Tensor,
    runs: int = LATENCY_RUNS,
    warmup: int = LATENCY_WARMUP,
) -> dict:
    # Micro-benchmark latency at batch sizes relevant to edge/online inference.
    model.eval()
    out = {}
    for batch_size in [1, 8, 32]:
        if x_test_t.size(0) < batch_size:
            continue
        samples = x_test_t[:batch_size]
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(samples)
            if samples.is_cuda:
                torch.cuda.synchronize()
        start = perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(samples)
            if samples.is_cuda:
                torch.cuda.synchronize()
        elapsed = perf_counter() - start
        mean_ms = (elapsed / runs) * 1000.0
        out[f"batch_{batch_size}"] = {
            "mean_ms_per_batch": float(mean_ms),
            "mean_ms_per_sample": float(mean_ms / batch_size),
            "runs": int(runs),
            "warmup_runs": int(warmup),
        }
    return out


def apply_global_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Apply global unstructured L1 pruning to Linear weights and make it permanent."""
    params_to_prune = [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]
    if not params_to_prune:
        return model

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, param_name in params_to_prune:
        prune.remove(module, param_name)
    return model


def run_pruning_sweep(
    model: nn.Module,
    x_test_t: torch.Tensor,
    y_test: np.ndarray,
    threshold: float,
) -> list[dict]:
    # Evaluate accuracy-efficiency tradeoff at several pruning levels.
    results = []
    for ratio in PRUNING_RATIOS:
        pruned_model = deepcopy(model)
        pruned_model = apply_global_pruning(pruned_model, amount=ratio)

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
    # Full per-scenario flow: split -> preprocess -> train -> tune -> evaluate -> profile.
    x_train_raw, y_train, x_test_raw, y_test, dropped = prepare_xy(train_df, test_df, DROP_COLS)

    x_train_fit_raw, x_val_raw, y_train_fit, y_val = train_test_split(
        x_train_raw,
        y_train,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=y_train.astype(int),
    )
    # Validation fold is only for threshold tuning; test remains untouched for final metrics.
    x_train, x_val, x_test, preprocessor_pipe = preprocess(x_train_fit_raw, x_val_raw, x_test_raw)
    contamination_report = check_train_test_overlap(x_train, x_test)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_fit, dtype=torch.float32, device=DEVICE).view(-1, 1)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    # Compute class weight only from the training fold (not val/test).
    positive = int(np.sum(y_train_fit == 1))
    negative = int(np.sum(y_train_fit == 0))
    if USE_CLASS_IMBALANCE_HANDLING and positive > 0:
        pos_weight_t = torch.tensor([negative / max(1, positive)], dtype=torch.float32, device=DEVICE)
    else:
        pos_weight_t = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)

    model = SimpleANN(input_dim=x_train.shape[1]).to(DEVICE)
    print(
        f"\n[{scenario_name}] data: train={x_train.shape[0]} | val={x_val.shape[0]} | "
        f"test={x_test.shape[0]} | features={x_train.shape[1]} | pos_weight={float(pos_weight_t.item()):.2f}"
    )
    losses = train_model(
        model,
        x_train_t,
        y_train_t,
        pos_weight_t=pos_weight_t,
        scenario_name=scenario_name,
    )

    threshold_source = "fallback"
    threshold = SCENARIO_THRESHOLD_FALLBACKS.get(scenario_name, SCENARIO_THRESHOLD_FALLBACKS["default"])
    val_metrics = evaluate_with_threshold(model, x_val_t, y_val, threshold=threshold)
    threshold_warning: str | None = None
    if USE_THRESHOLD_TUNING:
        tuned_threshold, tuned_val_metrics, threshold_warning = tune_threshold(model, x_val_t, y_val)
        threshold = tuned_threshold
        val_metrics = tuned_val_metrics
        threshold_source = "validation_tuned"

    # Final baseline metrics are always computed on the scenario test split.
    baseline_metrics = evaluate_with_threshold(model, x_test_t, y_test, threshold=threshold)
    pruning_results: list[dict] = []
    if ENABLE_PRUNING:
        pruning_results = run_pruning_sweep(model, x_test_t, y_test, threshold=threshold)

    model_stats = model_size_stats(model)
    sparsity = sparsity_stats(model)
    latency = latency_benchmark(model, x_test_t)
    batch_1 = latency.get("batch_1", {})

    print(f"[{scenario_name}] baseline ({threshold_source}, th={threshold:.2f}): {format_metrics(baseline_metrics)}")
    if threshold_warning:
        print(f"[{scenario_name}] WARNING: {threshold_warning}")
    if contamination_report["warning"]:
        print(f"[{scenario_name}] WARNING: {contamination_report['warning']}")
    if ENABLE_PRUNING:
        for pr in pruning_results:
            ratio_pct = int(pr["target_pruning_ratio"] * 100)
            m = pr["metrics"]
            print(f"[{scenario_name}] prune {ratio_pct:>2}%: {format_metrics(m)}")

    model_path = None
    preprocessor_path = None
    if SAVE_MODEL:
        model_path = OUTPUT_DIR / f"{scenario_name}_model.pt"
        preprocessor_path = OUTPUT_DIR / f"{scenario_name}_preprocessor.pkl"
        torch.save(model.state_dict(), model_path)
        joblib.dump(preprocessor_pipe, preprocessor_path)
        print(f"[{scenario_name}] saved model: {model_path}")
        print(f"[{scenario_name}] saved preprocessor: {preprocessor_path}")

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
        "positive_class_weight": float(pos_weight_t.item()),
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
        "threshold_warning": threshold_warning,
        "train_test_overlap": contamination_report,
        "validation_metrics": val_metrics,
        "last_epoch_loss": float(losses[-1]),
        "baseline_metrics": baseline_metrics,
        "inference_time_ms_per_sample": float(batch_1.get("mean_ms_per_sample", 0.0)),
        "model_stats": model_stats,
        "sparsity": sparsity,
        "latency": latency,
        "pruning_sweep": pruning_results,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "preprocessor_path": str(preprocessor_path) if preprocessor_path else None,
        },
    }


def main() -> None:
    global BATCH_SIZE, EPOCHS, LR, SEED, DATA_DIR, OUTPUT_DIR, RESULTS_JSON, QUIET, ENABLE_PRUNING, SAVE_MODEL

    args = parse_args()
    selected_scenarios = (
        list(SCENARIO_FILES.keys()) if "all" in args.scenario else list(dict.fromkeys(args.scenario))
    )
    DATA_DIR = Path(args.data_dir)
    RESULTS_JSON = Path(args.output_json)
    OUTPUT_DIR = RESULTS_JSON.parent
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    SEED = args.seed
    QUIET = args.quiet
    ENABLE_PRUNING = not args.no_pruning
    SAVE_MODEL = args.save_model
    datasets = resolve_datasets(DATA_DIR, selected_scenarios)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sanity_reports: dict[str, dict] = {}
    experiments: list[dict] = []

    print("\n=== IEC-61850 ANN experiment ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Scenarios: {', '.join(datasets.keys())}")
    print(f"Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LR}, Seed={SEED}\n")

    for name, split in datasets.items():
        print(f"=== Scenario: {name} ===")
        train_csv = split["train"]
        test_csv = split["test"]
        train_df, test_df = read_split(train_csv, test_csv)
        sanity = dataset_sanity_report(train_df, test_df)
        sanity_reports[name] = sanity

        print(
            f"Sanity [{name}] rows(train/test)={sanity['train_rows']}/{sanity['test_rows']} | "
            f"class_train={sanity['train_class_distribution']}"
        )

        result = run_experiment(
            scenario_name=name,
            train_df=train_df,
            test_df=test_df,
            train_csv=train_csv,
            test_csv=test_csv,
        )
        experiments.append(result)

    print("\n=== Final scenario summary ===")
    print(f"{'Scenario':<20}{'ACC':>8}{'F1':>8}{'MCC':>8}{'TH':>8}")
    print("-" * 52)
    for exp in experiments:
        m = exp["baseline_metrics"]
        print(
            f"{exp['scenario']:<20}"
            f"{m['accuracy']:>8.4f}"
            f"{m['f1']:>8.4f}"
            f"{m['mcc']:>8.4f}"
            f"{exp['decision_threshold']:>8.2f}"
        )

    payload = {
        # Single structured artifact for comparisons, plots, and reporting.
        "seed": SEED,
        "dataset_family": "IEC-61850 GOOSE",
        "data_dir": str(DATA_DIR),
        "scenarios": list(datasets.keys()),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LR),
        "quiet": bool(QUIET),
        "pruning_enabled": bool(ENABLE_PRUNING),
        "save_model": bool(SAVE_MODEL),
        "pruning_ratios": PRUNING_RATIOS,
        "threshold_candidates": [float(x) for x in THRESHOLD_CANDIDATES.tolist()],
        "sanity": sanity_reports,
        "experiments": experiments,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved baseline+pruning results to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
