from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from modeling.splits import stratified_split
from modeling.ereno_labels import (
    BINARY_LABELS,
    binary_counts_from_multiclass,
    format_binary_count_line,
    multiclass_label_to_binary_id,
    task_type_for_mode,
)
from modeling.binary_eval import (
    attack_scores_from_logits,
    compute_binary_metrics,
    format_binary_console_summary,
    multiclass_breakdown,
    write_binary_aggregate_summary_md,
    write_binary_summary_md,
)
from iec60.preprocess_ereno import apply_duplicate_removal, apply_variance_feature_filter
from iec60.tabular_guard import assert_comma_separated_csv
from modeling.runtime_metrics import inference_evaluation

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = BASE_DIR / "train.csv"
DEFAULT_TEST_CSV = BASE_DIR / "test.csv"
DEFAULT_OUTPUT_JSON = BASE_DIR / "outputs" / "ereno_ann_pruning_results.json"
DEFAULT_TARGET_COL = "class"
KNOWN_LABELS = {
    "normal",
    "random_replay",
    "inverse_replay",
    "masquerade_fake_fault",
    "masquerade_fake_normal",
    "injection",
    "high_StNum",
    "poisoned_high_rate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ERENO IEC-61850 multiclass ANN trainer (optional pruning / GA)")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument(
        "--use-official-test",
        action="store_true",
        help="Train on all of train-csv and evaluate on test-csv; otherwise stratified 80/20 split of train-csv only.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42, help="Used when --seeds is not set.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. 41,42,43). If set, run one full train+eval per seed and "
        "write mean±std for macro F1, MCC, etc. to output-json and summary-md.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="If set, write a short markdown table with baseline metrics (incl. MCC).",
    )
    parser.add_argument(
        "--pruning-ratios",
        type=str,
        default="",
        help="Comma-separated pruning amounts in [0,1), e.g. 0.2,0.5. Empty = no pruning sweep (baseline only).",
    )
    parser.add_argument("--use-ga-feature-selection", action="store_true")
    parser.add_argument("--ga-population", type=int, default=20)
    parser.add_argument("--ga-generations", type=int, default=10)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.05)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8)
    parser.add_argument("--ga-min-features", type=int, default=4)
    parser.add_argument("--ga-eval-epochs", type=int, default=4)
    parser.add_argument("--ga-sample-size", type=int, default=12000, help="0 means use all train rows for GA")
    parser.add_argument(
        "--no-drop-duplicates",
        action="store_true",
        help="Keep exact duplicate rows within each train/eval split (default: remove duplicates).",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.0,
        help="Drop numeric features with variance <= this value on the fit split only (default 0 = drop constants).",
    )
    parser.add_argument(
        "--skip-variance-filter",
        action="store_true",
        help="Do not apply variance-based feature dropping.",
    )
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each pipeline stage (with timings) and every training epoch; emit per-class report after eval.",
    )
    parser.add_argument(
        "--binary-mode",
        action="store_true",
        help="Binary IDS: normal=0, every other multiclass label=1. Multiclass remains the default.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def parse_seeds_csv(s: str | None) -> list[int] | None:
    if s is None or not str(s).strip():
        return None
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def write_summary_md_aggregate(
    path: Path,
    seeds: list[int],
    aggregate: dict[str, dict[str, float]],
    per_seed_rows: list[dict[str, Any]],
    *,
    binary_mode: bool = False,
) -> None:
    if binary_mode:
        write_binary_aggregate_summary_md(path, seeds, aggregate, per_seed_rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ERENO IEC-61850 ANN baseline (multi-seed)",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Mean ± std (test set — classification)",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("accuracy", "Accuracy"),
        ("mcc", "MCC"),
    ]:
        m = aggregate[key]
        lines.append(f"| {label} | {m['mean']:.4f} | {m['std']:.4f} |")

    opt_rows = [
        ("inference_ms_per_sample", "Inference ms/sample (canonical)"),
        ("cpu_rss_peak_mb", "CPU RSS peak (MB, process)"),
        ("gpu_peak_reserved_mb", "GPU peak reserved (MB)"),
    ]
    extra_block = any(k in aggregate for k, _ in opt_rows)
    if extra_block:
        lines.extend(["", "## Mean ± std (inference / memory)", "", "| Metric | Mean | Std |", "|---|---:|---:|"])
        for key, label in opt_rows:
            if key not in aggregate:
                continue
            m = aggregate[key]
            lines.append(f"| {label} | {m['mean']:.4f} | {m['std']:.4f} |")

    lines.extend(
        [
            "",
            "## Per seed (classification + inference + memory)",
            "",
            "| Seed | Macro F1 | Acc | MCC | ms/sample | CPU RSS peak MB | GPU peak MB | Model MB (FP32 est.) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed_rows:
        met = row["metrics"]
        ms = row.get("inference_ms_per_sample")
        ms_s = f"{ms:.4f}" if ms is not None else "n/a"
        rss = row.get("cpu_rss_peak_mb")
        rss_s = f"{rss:.1f}" if rss is not None else "n/a"
        gpu = row.get("gpu_peak_reserved_mb")
        gpu_s = f"{gpu:.1f}" if gpu is not None else "n/a"
        sz = row.get("model_size_mb_fp32")
        sz_s = f"{sz:.2f}" if sz is not None else "n/a"
        lines.append(
            f"| {row['seed']} | {met['macro_f1']:.4f} | {met['accuracy']:.4f} | {met['mcc']:.4f} | "
            f"{ms_s} | {rss_s} | {gpu_s} | {sz_s} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_optional_floats(rows: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    vals = [float(r[field]) for r in rows if r.get(field) is not None and np.isfinite(r[field])]
    return _mean_std(vals) if vals else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


PIPELINE_STEPS = 13


def _pipeline_log(args: argparse.Namespace, step: int, msg: str, t0: float | None = None) -> None:
    if args.quiet:
        return
    extra = ""
    if t0 is not None and getattr(args, "verbose", False):
        extra = f" (+{perf_counter() - t0:.2f}s)"
    print(f"[pipeline {step:02d}/{PIPELINE_STEPS}]{extra} {msg}", flush=True)


def recover_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset.")
    if df[target_col].notna().any():
        return df
    if "delay" not in df.columns:
        return df

    candidate = df["delay"].astype(str).str.strip()
    if not candidate.isin(KNOWN_LABELS).any():
        return df

    out = df.copy()
    out[target_col] = candidate
    out = out.drop(columns=["delay"], errors="ignore")
    print("Recovered target labels from 'delay' column because target column was empty.")
    return out


def drop_identifier_columns(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    lowered = {c.lower(): c for c in df.columns}
    dropped: list[str] = []
    for k in ("id", "flow_id", "sample_id", "record_id"):
        if k in lowered and lowered[k] != target_col:
            dropped.append(lowered[k])
    if not dropped:
        return df, []
    return df.drop(columns=dropped, errors="ignore"), dropped


def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    binary_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    # Avoid `.copy()` here: on ~3M rows a deep copy duplicates object + numeric blocks
    # and commonly OOMs; this function only reads columns into new numpy arrays below.
    x_train = train_df.drop(columns=[target_col], errors="ignore")
    x_test = test_df.drop(columns=[target_col], errors="ignore")
    y_train_labels = train_df[target_col].astype(str).to_numpy()
    y_test_labels = test_df[target_col].astype(str).to_numpy()

    if binary_mode:
        labels = list(BINARY_LABELS)
        y_train = np.array([multiclass_label_to_binary_id(x) for x in y_train_labels], dtype=np.int64)
        y_test = np.array([multiclass_label_to_binary_id(x) for x in y_test_labels], dtype=np.int64)
    else:
        labels = sorted(pd.Series(y_train_labels).unique().tolist())
        label_to_id = {lab: i for i, lab in enumerate(labels)}
        y_train = np.array([label_to_id[x] for x in y_train_labels], dtype=np.int64)
        y_test = np.array([label_to_id.get(x, -1) for x in y_test_labels], dtype=np.int64)
        unknown_mask = y_test == -1
        if unknown_mask.any():
            raise ValueError("Test set contains unseen classes not present in training set.")

    obj_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train.columns if c not in obj_cols]

    if num_cols:
        tr_num = x_train[num_cols].apply(pd.to_numeric, errors="coerce")
        te_num = x_test[num_cols].apply(pd.to_numeric, errors="coerce")
        med = tr_num.median(numeric_only=True)
        tr_num = tr_num.fillna(med).astype(np.float32)
        te_num = te_num.fillna(med).astype(np.float32)
        mean = tr_num.mean(axis=0)
        std = tr_num.std(axis=0).replace(0.0, 1.0)
        tr_num = ((tr_num - mean) / std).to_numpy(dtype=np.float32)
        te_num = ((te_num - mean) / std).to_numpy(dtype=np.float32)
    else:
        tr_num = np.empty((len(x_train), 0), dtype=np.float32)
        te_num = np.empty((len(x_test), 0), dtype=np.float32)

    if obj_cols:
        tr_cat_blocks: list[np.ndarray] = []
        te_cat_blocks: list[np.ndarray] = []
        for c in obj_cols:
            tr = x_train[c].astype(str).fillna("__NA__")
            te = x_test[c].astype(str).fillna("__NA__")
            values = sorted(tr.unique().tolist())
            mapping = {v: i for i, v in enumerate(values)}
            tr_enc = tr.map(mapping).fillna(-1).to_numpy(dtype=np.float32).reshape(-1, 1)
            te_enc = te.map(mapping).fillna(-1).to_numpy(dtype=np.float32).reshape(-1, 1)
            tr_cat_blocks.append(tr_enc)
            te_cat_blocks.append(te_enc)
        tr_cat = np.concatenate(tr_cat_blocks, axis=1).astype(np.float32)
        te_cat = np.concatenate(te_cat_blocks, axis=1).astype(np.float32)
    else:
        tr_cat = np.empty((len(x_train), 0), dtype=np.float32)
        te_cat = np.empty((len(x_test), 0), dtype=np.float32)

    x_train_mat = np.concatenate([tr_num, tr_cat], axis=1).astype(np.float32)
    x_test_mat = np.concatenate([te_num, te_cat], axis=1).astype(np.float32)
    feature_names = num_cols + [f"{c}__cat_code" for c in obj_cols]
    return x_train_mat, y_train, x_test_mat, y_test, labels, feature_names


class ErenoANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        dims = list(hidden_dims) if hidden_dims else [128, 64]
        if any(d <= 0 for d in dims):
            raise ValueError(f"hidden_dims must be positive ints, got {dims}.")
        self.hidden_dims = dims
        layers: list[nn.Module] = []
        prev = input_dim
        for h in dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer activations (output of the last ReLU)."""
        modules = list(self.net.children())
        for layer in modules[:-1]:
            x = layer(x)
        return x


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    class_weights: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    quiet: bool,
    *,
    epoch_log_stride: int = 3,
) -> list[float]:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    n = x_train.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=x_train.device)
        running = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        epoch_loss = running / max(1, batches)
        losses.append(epoch_loss)
        stride = max(1, int(epoch_log_stride))
        if (not quiet) and (epoch == 0 or (epoch + 1) % stride == 0 or (epoch + 1) == epochs):
            print(f"Epoch {epoch + 1:02d}/{epochs} loss={epoch_loss:.4f}")
    return losses


def evaluate_model(
    model: nn.Module,
    x_eval: torch.Tensor,
    y_eval_np: np.ndarray,
    *,
    num_classes: int | None = None,
    binary_mode: bool = False,
    y_true_multiclass: np.ndarray | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(x_eval)
        pred = logits.argmax(dim=1).cpu().numpy()
    if binary_mode and num_classes == 2:
        scores = attack_scores_from_logits(logits.cpu().numpy())
        metrics = compute_binary_metrics(y_eval_np, pred, scores)
        metrics["weighted_f1"] = metrics["macro_f1"]
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_eval_np, pred))
        if y_true_multiclass is not None and len(y_true_multiclass) == len(y_eval_np):
            metrics["multiclass_breakdown"] = multiclass_breakdown(
                y_eval_np, pred, np.asarray(y_true_multiclass, dtype=object)
            )
        return metrics
    f1_kw: dict[str, Any] = {"zero_division": 0}
    if num_classes is not None and num_classes > 0:
        f1_kw["labels"] = np.arange(num_classes)
    return {
        "macro_f1": float(f1_score(y_eval_np, pred, average="macro", **f1_kw)),
        "weighted_f1": float(f1_score(y_eval_np, pred, average="weighted", **f1_kw)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval_np, pred)),
        "accuracy": float(accuracy_score(y_eval_np, pred)),
        "mcc": float(matthews_corrcoef(y_eval_np, pred)),
    }


def write_summary_md(
    path: Path,
    metrics: dict,
    inference: dict[str, Any] | None,
    model_stats: dict | None,
    *,
    binary_mode: bool = False,
    dataset_info: dict[str, Any] | None = None,
    seed: int | None = None,
) -> None:
    if binary_mode or "tn" in metrics:
        write_binary_summary_md(
            path,
            metrics,
            inference=inference,
            model_stats=model_stats,
            dataset_info=dataset_info,
            seed=seed,
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ERENO IEC-61850 ANN evaluation (multiclass)",
        "",
        "## Classification (test set)",
        "",
        "| Macro F1 | Weighted F1 | Balanced Acc | Acc | MCC |",
        "|---:|---:|---:|---:|---:|",
        "| "
        f"{metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} | "
        f"{metrics['balanced_accuracy']:.4f} | {metrics['accuracy']:.4f} | {metrics['mcc']:.4f} |",
        "",
    ]
    if inference is not None:
        ms = inference.get("canonical_inference_ms_per_sample")
        ms_s = f"{ms:.4f}" if ms is not None else "n/a"
        mem = inference.get("memory") or {}
        rss = mem.get("cpu_rss_mb_peak_during_inference")
        rss_s = f"{rss:.1f}" if rss is not None else "n/a"
        gpu = mem.get("gpu_peak_reserved_mb")
        gpu_s = f"{gpu:.1f}" if gpu is not None else "n/a"
        sz = (model_stats or {}).get("estimated_size_mb_fp32")
        sz_s = f"{sz:.2f}" if sz is not None else "n/a"
        lines.extend(
            [
                "## Inference and resources",
                "",
                "| ms/sample (canonical batch) | Model size (MB, FP32 est.) | CPU RSS peak (MB) | GPU peak reserved (MB) |",
                "|---:|---:|---:|---:|",
                f"| {ms_s} | {sz_s} | {rss_s} | {gpu_s} |",
                "",
            ]
        )
        if mem.get("notes"):
            lines.append("Notes: " + " ".join(mem["notes"]))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def model_size_stats(model: nn.Module) -> dict:
    params = sum(p.numel() for p in model.parameters())
    return {
        "total_params": int(params),
        "estimated_size_mb_fp32": float((params * 4) / (1024 ** 2)),
    }


def sparsity_stats(model: nn.Module) -> dict:
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.detach()
            total += int(w.numel())
            zeros += int((w == 0).sum().item())
    return {
        "total_weight_params": int(total),
        "zero_weight_params": int(zeros),
        "sparsity_ratio": float(zeros / max(1, total)),
    }


def latency_benchmark(model: nn.Module, x_eval: torch.Tensor, runs: int = 200, warmup: int = 20) -> dict[str, Any]:
    """Per-batch-size latency; optional process RSS peak (psutil) and GPU peak reserved (CUDA)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    proc = psutil.Process() if psutil is not None else None

    model.eval()
    out: dict[str, Any] = {}
    rss_peak_mb: float | None = None

    dev = x_eval.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    for bs in (1, 8, 32):
        if x_eval.size(0) < bs:
            continue
        sample = x_eval[:bs]
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample)
                if proc is not None:
                    rss_peak_mb = max(
                        rss_peak_mb or 0.0,
                        float(proc.memory_info().rss / (1024**2)),
                    )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
        t0 = perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(sample)
                if proc is not None:
                    rss_peak_mb = max(
                        rss_peak_mb or 0.0,
                        float(proc.memory_info().rss / (1024**2)),
                    )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
        ms = ((perf_counter() - t0) / runs) * 1000.0
        out[f"batch_{bs}"] = {
            "mean_ms_per_batch": float(ms),
            "mean_ms_per_sample": float(ms / bs),
        }

    if proc is not None and rss_peak_mb is not None:
        out["process_rss_peak_mb"] = float(rss_peak_mb)
    if dev.type == "cuda":
        out["gpu_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(dev) / (1024**2))
    return out


def apply_global_pruning(model: nn.Module, amount: float) -> nn.Module:
    params = [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for m, name in params:
        prune.remove(m, name)
    return model


def parse_pruning_ratios(ratios: str) -> list[float]:
    out: list[float] = []
    for x in ratios.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if v < 0.0 or v >= 1.0:
            raise ValueError("Pruning ratios must be in [0.0, 1.0).")
        out.append(v)
    return out


def _ensure_min_features(mask: np.ndarray, min_features: int) -> np.ndarray:
    m = mask.copy()
    if int(m.sum()) >= min_features:
        return m
    missing = min_features - int(m.sum())
    zero_idx = np.where(~m)[0]
    if zero_idx.size == 0:
        return m
    pick = np.random.choice(zero_idx, size=min(missing, zero_idx.size), replace=False)
    m[pick] = True
    return m


def _evaluate_mask_fitness(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mask: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_classes: int,
    *,
    rare_class_ids: list[int] | None = None,
    rare_bonus: float = 0.0,
) -> float:
    xtr = torch.tensor(x_train[:, mask], dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    xva = torch.tensor(x_val[:, mask], dtype=torch.float32, device=device)
    cls_idx = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=cls_idx, y=y_train)
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    model = ErenoANN(input_dim=xtr.shape[1], num_classes=num_classes).to(device)
    train_model(
        model=model,
        x_train=xtr,
        y_train=ytr,
        class_weights=class_weights_t,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        quiet=True,
        epoch_log_stride=3,
    )
    model.eval()
    with torch.no_grad():
        pred = model(xva).argmax(dim=1).cpu().numpy()
    macro_f1 = float(
        f1_score(y_val, pred, average="macro", zero_division=0, labels=cls_idx)
    )
    sparsity_penalty = 0.01 * (mask.sum() / max(1, len(mask)))
    rare_recall_bonus = 0.0
    if rare_class_ids and rare_bonus > 0.0:
        recall_sum = 0.0
        for cid in rare_class_ids:
            true_mask = y_val == cid
            n = int(true_mask.sum())
            if n > 0:
                recall_sum += float((pred[true_mask] == cid).sum()) / n
        rare_recall_bonus = float(rare_bonus) * recall_sum
    return float(macro_f1 - sparsity_penalty + rare_recall_bonus)


def run_ga_feature_selection(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    if x_train_np.shape[1] <= args.ga_min_features:
        mask = np.ones(x_train_np.shape[1], dtype=bool)
        return mask, {"enabled": True, "note": "Feature count already <= ga_min_features."}

    x_source = x_train_np
    y_source = y_train_np
    if args.ga_sample_size > 0 and len(y_train_np) > args.ga_sample_size:
        idx = np.random.choice(len(y_train_np), size=args.ga_sample_size, replace=False)
        x_source = x_train_np[idx]
        y_source = y_train_np[idx]

    counts = np.bincount(y_source)
    can_stratify = bool(np.all(counts[counts > 0] >= 2))
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_source,
        y_source,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_source if can_stratify else None,
    )
    feature_count = x_train_np.shape[1]
    pop_size = max(4, args.ga_population)
    raw_protected = list(getattr(args, "ga_protected_indices", []) or [])
    protected_indices = sorted({int(i) for i in raw_protected if 0 <= int(i) < feature_count})

    def _force_protected(mask: np.ndarray) -> np.ndarray:
        if protected_indices:
            mask[protected_indices] = True
        return mask

    population: list[np.ndarray] = []
    for _ in range(pop_size):
        chrom = np.random.rand(feature_count) < 0.5
        chrom = _force_protected(chrom)
        chrom = _ensure_min_features(chrom, args.ga_min_features)
        population.append(chrom)

    history: list[dict] = []
    best_mask = population[0]
    best_score = -1e9
    rare_class_ids = list(getattr(args, "ga_rare_class_ids", []) or [])
    rare_bonus = float(getattr(args, "ga_rare_bonus", 0.0) or 0.0)
    for gen in range(args.ga_generations):
        scores = [
            _evaluate_mask_fitness(
                x_fit,
                y_fit,
                x_val,
                y_val,
                mask=chrom,
                epochs=args.ga_eval_epochs,
                batch_size=max(64, min(args.batch_size, 512)),
                lr=args.lr,
                device=device,
                num_classes=len(np.unique(y_train_np)),
                rare_class_ids=rare_class_ids,
                rare_bonus=rare_bonus,
            )
            for chrom in population
        ]
        best_idx = int(np.argmax(scores))
        gen_best_score = float(scores[best_idx])
        gen_best_mask = population[best_idx].copy()
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_mask = gen_best_mask

        history.append(
            {
                "generation": gen + 1,
                "best_fitness": gen_best_score,
                "selected_features": int(gen_best_mask.sum()),
            }
        )
        print(
            f"[GA] generation {gen + 1}/{args.ga_generations}: "
            f"best_fitness={gen_best_score:.4f}, selected={int(gen_best_mask.sum())}"
        )

        elite_count = max(2, pop_size // 4)
        elite_idx = np.argsort(scores)[-elite_count:]
        elites = [population[i].copy() for i in elite_idx]
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(elites), random.choice(elites)
            if random.random() < args.ga_crossover_rate:
                point = random.randint(1, feature_count - 1)
                child = np.concatenate([p1[:point], p2[point:]])
            else:
                child = p1.copy()
            mut = np.random.rand(feature_count) < args.ga_mutation_rate
            child[mut] = ~child[mut]
            child = _force_protected(child)
            child = _ensure_min_features(child, args.ga_min_features)
            new_pop.append(child)
        population = new_pop[:pop_size]

    selected_names = [feature_names[i] for i in np.where(best_mask)[0]]
    protected_names = [feature_names[i] for i in protected_indices]
    return best_mask, {
        "enabled": True,
        "population": int(pop_size),
        "generations": int(args.ga_generations),
        "min_features": int(args.ga_min_features),
        "eval_epochs": int(args.ga_eval_epochs),
        "sample_rows_for_ga": int(len(y_source)),
        "best_fitness": float(best_score),
        "selected_feature_count": int(best_mask.sum()),
        "selected_features": selected_names,
        "protected_feature_indices": protected_indices,
        "protected_feature_names": protected_names,
        "rare_bonus_aggregate": "sum",
        "history": history,
    }


def run_training(args: argparse.Namespace, *, write_artifacts: bool = True) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_run = perf_counter()

    if not args.train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    _pipeline_log(args, 1, f"Device={device}; validate CSV paths", t_run)
    assert_comma_separated_csv(args.train_csv, role="Train")
    assert_comma_separated_csv(args.test_csv, role="Test")

    _pipeline_log(args, 2, "Loading train/test into memory (pandas.read_csv)", t_run)
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    _pipeline_log(
        args,
        3,
        f"Loaded tables: train_rows={len(train_df)} test_rows={len(test_df)} columns={train_df.shape[1]}",
        t_run,
    )

    if args.target_col not in train_df.columns or args.target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' must exist in both datasets. "
            f"train_has={args.target_col in train_df.columns}, test_has={args.target_col in test_df.columns}"
        )

    _pipeline_log(args, 4, f"Target column={args.target_col!r}; recover / align labels if needed", t_run)
    train_df = recover_target_column(train_df, args.target_col)
    test_df = recover_target_column(test_df, args.target_col)

    if args.max_rows > 0:
        train_df = train_df.head(args.max_rows).copy()
        if args.use_official_test:
            test_df = test_df.head(args.max_rows).copy()
        _pipeline_log(args, 5, f"Applied --max-rows={args.max_rows}", t_run)
    else:
        _pipeline_log(args, 5, "No --max-rows cap (using full tables)", t_run)

    if args.use_official_test:
        fit_df = train_df
        eval_df = test_df
        _pipeline_log(args, 6, "Split: using official train vs test (no row split)", t_run)
    else:
        fit_df, eval_df = stratified_split(train_df, target_col=args.target_col, random_state=args.seed)
        _pipeline_log(args, 6, "Split: stratified 80/20 from train CSV", t_run)

    fit_df = fit_df.dropna(subset=[args.target_col]).reset_index(drop=True)
    eval_df = eval_df.dropna(subset=[args.target_col]).reset_index(drop=True)
    _pipeline_log(
        args,
        7,
        f"After dropna: fit_rows={len(fit_df)} eval_rows={len(eval_df)}",
        t_run,
    )

    fit_df, dropped_fit = drop_identifier_columns(fit_df, args.target_col)
    eval_df, dropped_eval = drop_identifier_columns(eval_df, args.target_col)
    dropped = sorted(set(dropped_fit + dropped_eval))
    if dropped:
        print(f"Dropped identifier columns: {dropped}")
    _pipeline_log(args, 8, f"Identifier drop complete; dropped={dropped or 'none'}", t_run)

    preprocess_summary: dict[str, Any] = {
        "drop_duplicates_enabled": not bool(getattr(args, "no_drop_duplicates", False)),
        "variance_filter_enabled": not bool(getattr(args, "skip_variance_filter", False)),
        "variance_threshold": float(getattr(args, "variance_threshold", 0.0)),
        "duplicates_removed_fit": 0,
        "duplicates_removed_eval": 0,
        "variance_dropped_columns": [],
    }
    if not getattr(args, "no_drop_duplicates", False):
        fit_df, n_dup_fit = apply_duplicate_removal(fit_df)
        eval_df, n_dup_eval = apply_duplicate_removal(eval_df)
        preprocess_summary["duplicates_removed_fit"] = n_dup_fit
        preprocess_summary["duplicates_removed_eval"] = n_dup_eval
        if n_dup_fit or n_dup_eval:
            print(f"Removed duplicate rows: fit={n_dup_fit}, eval={n_dup_eval}")

    if not getattr(args, "skip_variance_filter", False):
        thr = float(getattr(args, "variance_threshold", 0.0))
        fit_df, eval_df, var_dropped = apply_variance_feature_filter(fit_df, eval_df, args.target_col, thr)
        preprocess_summary["variance_dropped_columns"] = var_dropped
        if var_dropped:
            print(f"Variance filter (fit-only, threshold={thr}): dropped {len(var_dropped)} columns: {var_dropped[:12]}{'...' if len(var_dropped) > 12 else ''}")
    _pipeline_log(args, 9, "Numeric impute + scaling + categorical encoding (encode_features)", t_run)

    binary_mode = bool(getattr(args, "binary_mode", False))
    x_train_np, y_train_np, x_test_np, y_test_np, labels, feature_names = encode_features(
        fit_df, eval_df, args.target_col, binary_mode=binary_mode
    )
    _pipeline_log(
        args,
        10,
        f"Encoded: X_train={x_train_np.shape} X_test={x_test_np.shape} labels={len(labels)}",
        t_run,
    )
    ga_summary = {"enabled": False}
    if args.use_ga_feature_selection:
        _pipeline_log(args, 11, "Genetic algorithm feature selection", t_run)
        if getattr(args, "verbose", False) and not args.quiet:
            print(
                f"[GA] setup: raw_features={x_train_np.shape[1]} generations={args.ga_generations} "
                f"population={args.ga_population} eval_epochs={args.ga_eval_epochs} "
                f"sample_cap={args.ga_sample_size or 'all'}",
                flush=True,
            )
        ga_mask, ga_summary = run_ga_feature_selection(x_train_np, y_train_np, feature_names, args, device)
        x_train_np = x_train_np[:, ga_mask]
        x_test_np = x_test_np[:, ga_mask]
        feature_names = [feature_names[i] for i in np.where(ga_mask)[0]]
        print(f"[GA] using {x_train_np.shape[1]} selected features for final ANN training.")
    else:
        _pipeline_log(args, 11, "GA feature selection disabled", t_run)
    _pipeline_log(args, 12, "Final ANN: class weights + Adam training", t_run)
    print(f"[fit] rows={len(y_train_np)} | feature_columns={x_train_np.shape[1]} | classes={len(labels)}")
    print(f"[eval] rows={len(y_test_np)} | feature_columns={x_test_np.shape[1]} | classes={len(labels)}")
    print(f"[fit] class_distribution={fit_df[args.target_col].astype(str).value_counts().to_dict()}")
    print(f"[eval] class_distribution={eval_df[args.target_col].astype(str).value_counts().to_dict()}")
    if binary_mode:
        fit_mc = fit_df[args.target_col].astype(str).value_counts().to_dict()
        eval_mc = eval_df[args.target_col].astype(str).value_counts().to_dict()
        print(f"[binary] fit  {format_binary_count_line(binary_counts_from_multiclass(fit_mc))}", flush=True)
        print(f"[binary] eval {format_binary_count_line(binary_counts_from_multiclass(eval_mc))}", flush=True)

    x_train_t = torch.tensor(x_train_np, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_np, dtype=torch.long, device=device)
    x_test_t = torch.tensor(x_test_np, dtype=torch.float32, device=device)

    classes_idx = np.arange(len(labels))
    weights = compute_class_weight(class_weight="balanced", classes=classes_idx, y=y_train_np)
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = ErenoANN(input_dim=x_train_np.shape[1], num_classes=len(labels)).to(device)
    epoch_stride = 1 if getattr(args, "verbose", False) else 3
    effective_quiet = bool(args.quiet) and not getattr(args, "verbose", False)
    losses = train_model(
        model=model,
        x_train=x_train_t,
        y_train=y_train_t,
        class_weights=class_weights_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quiet=effective_quiet,
        epoch_log_stride=epoch_stride,
    )

    _pipeline_log(args, 13, "Evaluate on held-out set + latency / memory (post-train)", t_run)
    y_test_multiclass = eval_df[args.target_col].astype(str).values
    baseline_metrics = evaluate_model(
        model,
        x_test_t,
        y_test_np,
        num_classes=len(labels),
        binary_mode=binary_mode,
        y_true_multiclass=y_test_multiclass if binary_mode else None,
    )
    baseline_size = model_size_stats(model)
    baseline_sparsity = sparsity_stats(model)
    baseline_inference = inference_evaluation(model, x_test_t, device)
    ms_inf = baseline_inference.get("canonical_inference_ms_per_sample")
    ms_s = f"{ms_inf:.4f}" if ms_inf is not None else "n/a"
    rss = (baseline_inference.get("memory") or {}).get("cpu_rss_mb_peak_during_inference")
    rss_s = f"{rss:.1f}MB" if rss is not None else "n/a"
    gpu = (baseline_inference.get("memory") or {}).get("gpu_peak_reserved_mb")
    gpu_s = f"{gpu:.1f}MB" if gpu is not None else "n/a"
    sz_mb = baseline_size.get("estimated_size_mb_fp32")
    if binary_mode:
        print(f"baseline (binary): {format_binary_console_summary(baseline_metrics)} | infer_ms/sample={ms_s}, model_fp32~{sz_mb:.2f}MB, rss_peak~{rss_s}, gpu_peak~{gpu_s}")
    else:
        print(
            f"baseline: macro_f1={baseline_metrics['macro_f1']:.4f}, "
            f"weighted_f1={baseline_metrics['weighted_f1']:.4f}, "
            f"bal_acc={baseline_metrics['balanced_accuracy']:.4f}, acc={baseline_metrics['accuracy']:.4f}, "
            f"mcc={baseline_metrics['mcc']:.4f} | "
            f"infer_ms/sample={ms_s}, model_fp32~{sz_mb:.2f}MB, rss_peak~{rss_s}, gpu_peak~{gpu_s}"
        )
    if getattr(args, "verbose", False) and not args.quiet:
        model.eval()
        with torch.no_grad():
            pred_idx = model(x_test_t).argmax(dim=1).cpu().numpy()
        print("[pipeline] Per-class classification_report (test set):\n")
        print(
            classification_report(
                y_test_np,
                pred_idx,
                labels=np.arange(len(labels)),
                target_names=list(labels),
                zero_division=0,
            )
        )
    if getattr(args, "verbose", False) and not args.quiet:
        print(f"[pipeline] Total wall time from run start: {perf_counter() - t_run:.2f}s", flush=True)

    pruning_ratios = parse_pruning_ratios(args.pruning_ratios or "")
    pruning_results: list[dict] = []
    for ratio in pruning_ratios:
        m = deepcopy(model)
        apply_global_pruning(m, amount=ratio)
        metrics = evaluate_model(
            m,
            x_test_t,
            y_test_np,
            num_classes=len(labels),
            binary_mode=binary_mode,
            y_true_multiclass=y_test_multiclass if binary_mode else None,
        )
        size = model_size_stats(m)
        sparsity = sparsity_stats(m)
        inf = inference_evaluation(m, x_test_t, device)
        pruning_results.append(
            {
                "target_pruning_ratio": float(ratio),
                "metrics": metrics,
                "model_stats": size,
                "sparsity": sparsity,
                "inference": inf,
            }
        )
        if binary_mode:
            print(f"prune {int(ratio * 100):>2d}%: {format_binary_console_summary(metrics)}")
        else:
            print(
                f"prune {int(ratio * 100):>2d}%: macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, bal_acc={metrics['balanced_accuracy']:.4f}, "
                f"mcc={metrics['mcc']:.4f}"
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    fit_mc_counts = fit_df[args.target_col].astype(str).value_counts().to_dict()
    eval_mc_counts = eval_df[args.target_col].astype(str).value_counts().to_dict()
    payload = {
        "task_type": task_type_for_mode(binary_mode),
        "binary_mode": binary_mode,
        "dataset": {
            "train_csv": str(args.train_csv),
            "test_csv": str(args.test_csv),
            "target_col": args.target_col,
            "use_official_test": bool(args.use_official_test),
            "dropped_identifier_columns": dropped,
            "preprocessing": preprocess_summary,
            "train_rows": int(len(y_train_np)),
            "test_rows": int(len(y_test_np)),
            "feature_columns_after_encoding": int(x_train_np.shape[1]),
            "labels": labels,
            "multiclass_train_counts": {str(k): int(v) for k, v in fit_mc_counts.items()},
            "multiclass_eval_counts": {str(k): int(v) for k, v in eval_mc_counts.items()},
            "train_binary_class_counts": binary_counts_from_multiclass(fit_mc_counts) if binary_mode else None,
            "eval_binary_class_counts": binary_counts_from_multiclass(eval_mc_counts) if binary_mode else None,
            "final_feature_names": feature_names,
        },
        "ga_feature_selection": ga_summary,
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "device": str(device),
            "losses": losses,
        },
        "baseline": {
            "metrics": baseline_metrics,
            "model_stats": baseline_size,
            "sparsity": baseline_sparsity,
            "inference": baseline_inference,
        },
        "pruning_sweep": pruning_results,
    }
    if write_artifacts:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if args.summary_md is not None:
            dataset_info = {
                "train_rows": int(len(y_train_np)),
                "test_rows": int(len(y_test_np)),
                "train_binary_class_counts": binary_counts_from_multiclass(fit_mc_counts) if binary_mode else None,
                "eval_binary_class_counts": binary_counts_from_multiclass(eval_mc_counts) if binary_mode else None,
            }
            write_summary_md(
                args.summary_md,
                baseline_metrics,
                baseline_inference,
                baseline_size,
                binary_mode=binary_mode,
                dataset_info=dataset_info if binary_mode else None,
                seed=int(args.seed),
            )

        print(f"Saved results: {args.output_json}")
        if args.summary_md is not None:
            print(f"Saved summary: {args.summary_md}")

    if getattr(args, "save_model", False):
        model_path = args.output_json.with_name(f"{args.output_json.stem}_seed{args.seed}.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(x_train_np.shape[1]),
                "num_classes": int(len(labels)),
                "labels": list(labels),
                "seed": int(args.seed),
            },
            model_path,
        )
        print(f"Saved model checkpoint: {model_path}")
    return payload


def run_multi_seed_training(args: argparse.Namespace) -> dict:
    seeds = parse_seeds_csv(args.seeds)
    if seeds is None:
        raise ValueError("run_multi_seed_training requires args.seeds to be a non-empty comma-separated list.")
    runs_out: list[dict[str, Any]] = []
    last_payload: dict | None = None
    for seed in seeds:
        dup = deepcopy(args)
        dup.seed = int(seed)
        print(f"\n=== Seed {dup.seed} ===")
        p = run_training(dup, write_artifacts=False)
        last_payload = p
        inf = p["baseline"]["inference"]
        mem = inf.get("memory") or {}
        runs_out.append(
            {
                "seed": int(seed),
                "metrics": p["baseline"]["metrics"],
                "final_train_loss": float(p["training"]["losses"][-1]) if p["training"]["losses"] else None,
                "inference_ms_per_sample": inf.get("canonical_inference_ms_per_sample"),
                "cpu_rss_peak_mb": mem.get("cpu_rss_mb_peak_during_inference"),
                "cpu_percent_single_core_peak": mem.get("cpu_percent_single_core_peak_during_inference"),
                "gpu_peak_reserved_mb": mem.get("gpu_peak_reserved_mb"),
                "model_size_mb_fp32": p["baseline"]["model_stats"].get("estimated_size_mb_fp32"),
            }
        )

    binary_mode = bool(getattr(args, "binary_mode", False))
    if binary_mode:
        metric_keys = [
            ("attack_f1", "f1"),
            ("attack_recall", "recall"),
            ("attack_precision", "precision"),
            ("false_negative_rate", "false_negative_rate"),
            ("false_positive_rate", "false_positive_rate"),
            ("mcc", "mcc"),
            ("accuracy", "accuracy"),
        ]
    else:
        metric_keys = [
            ("macro_f1", "macro_f1"),
            ("weighted_f1", "weighted_f1"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("accuracy", "accuracy"),
            ("mcc", "mcc"),
        ]
    aggregate: dict[str, dict[str, float]] = {}
    for agg_key, src_key in metric_keys:
        vals = [float(r["metrics"][src_key]) for r in runs_out]
        aggregate[agg_key] = _mean_std(vals)

    for fld in (
        "inference_ms_per_sample",
        "cpu_rss_peak_mb",
        "cpu_percent_single_core_peak",
        "gpu_peak_reserved_mb",
        "model_size_mb_fp32",
    ):
        agg = _aggregate_optional_floats(runs_out, fld)
        if agg is not None:
            aggregate[fld] = agg

    combined: dict[str, Any] = {
        "experiment": "baseline_ann_multi_seed",
        "task_type": task_type_for_mode(bool(getattr(args, "binary_mode", False))),
        "binary_mode": bool(getattr(args, "binary_mode", False)),
        "seeds": seeds,
        "dataset": last_payload["dataset"] if last_payload else {},
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "use_official_test": bool(args.use_official_test),
        },
        "runs": runs_out,
        "aggregate": aggregate,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    if binary_mode:
        af1 = aggregate["attack_f1"]
        agg_line = (
            f"\nAggregate over seeds {seeds} (binary): "
            f"attack_F1={af1['mean']:.4f} +/- {af1['std']:.4f}, "
            f"mcc={aggregate['mcc']['mean']:.4f} +/- {aggregate['mcc']['std']:.4f}"
        )
    else:
        agg_line = (
            f"\nAggregate over seeds {seeds}: "
            f"macro_f1={aggregate['macro_f1']['mean']:.4f} +/- {aggregate['macro_f1']['std']:.4f}, "
            f"mcc={aggregate['mcc']['mean']:.4f} +/- {aggregate['mcc']['std']:.4f}"
        )
    if "inference_ms_per_sample" in aggregate:
        im = aggregate["inference_ms_per_sample"]
        agg_line += f", infer_ms/sample={im['mean']:.4f} +/- {im['std']:.4f}"
    print(agg_line)
    if args.summary_md is not None:
        write_summary_md_aggregate(args.summary_md, seeds, aggregate, runs_out, binary_mode=binary_mode)
        print(f"Saved summary: {args.summary_md}")
    print(f"Saved results: {args.output_json}")
    return combined


def run_from_parsed_args(args: argparse.Namespace) -> dict:
    """Dispatch single-seed or multi-seed training (used by CLI and by train_ereno.py)."""
    seeds = parse_seeds_csv(args.seeds)
    if seeds is not None and len(seeds) >= 1:
        return run_multi_seed_training(args)
    return run_training(args)


def main() -> None:
    run_from_parsed_args(parse_args())


if __name__ == "__main__":
    main()
