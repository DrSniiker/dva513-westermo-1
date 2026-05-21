"""
Full-file streaming ANN training for ERENO-sized CSVs (train + test read in chunks).

- **Coverage:** every row of `--train-csv` is visited for statistics + each epoch of training;
  every row of `--test-csv` is visited for evaluation (chunked).
- **Memory:** holds at most a few chunk-sized DataFrames plus a reservoir matrix for GA.
- **GA:** runs on a **reservoir sample** of encoded train rows (same GA code as train_ereno_ann).
  Default reservoir is **class-stratified** (proportional to train counts, min 1 per class when K ≥ C).
  Use `--uniform-ga-reservoir` for the old uniform global reservoir.
- **Loss:** optional **focal loss** via `--focal-gamma 2` (0 = plain cross-entropy with class weights).
  Use `--class-weights {balanced,sqrt,off}` (default `balanced`). Pair focal with `sqrt` or `off`
  to avoid stacking two rebalancers.
- **GA:** can target rare classes with `--ga-rare-classes name1,name2 --ga-rare-bonus 0.5`,
  which adds `bonus * mean_recall(rare)` to the fitness on top of macro F1 minus sparsity.
- **Pruning:** optional global L1 unstructured sweep after training (ratios in [0,1)).

Does **not** replicate exact duplicate-row removal across the full file (would need a second pass
with huge hash state). Use `--skip-variance-filter` if you want to match fewer preprocessing steps.

Binary baseline v1 (full ERENO, ~3M rows):
  .\\scripts\\run_ereno_binary_baseline_v1.ps1

Or manually:
  python src/modeling/train_ereno_stream.py \\
    --binary-mode --focal-gamma 2 --class-weights off --hidden-dims 256,128 \\
    --ga-protect-features timeFromLastChange,cbStatus,delay,stDiff,SqNum,gooseLen,APDUSize,timestampDiff \\
    --ga-rare-bonus 0.5 --save-model --save-best-pruned \\
    --output-json reports/binary_baseline_v1.json \\
    --artifacts-dir reports/binary_baseline_v1_artifacts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modeling.binary_eval import (
    attack_scores_from_logits,
    compute_binary_metrics,
    format_binary_console_summary,
    pick_best_pruning_sweep,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_tsne_binary,
    print_epoch_eval_report,
    print_stream_run_footer,
    save_selected_features,
    write_stream_run_summary_md,
)
from modeling.ereno_labels import (
    ATTACK_CLASS_NAME,
    NORMAL_CLASS_NAME,
    binary_counts_from_multiclass,
    count_binary_labels_in_csv,
    format_binary_count_line,
    multiclass_label_to_binary_id,
    stream_binary_state_fields,
    task_type_for_mode,
)
from iec60.paths import BINARY_BASELINE_ARTIFACTS_DIR, BINARY_BASELINE_JSON, STREAM_TEST_CSV, STREAM_TRAIN_CSV
from iec60.tabular_guard import assert_comma_separated_csv
from modeling.train_ereno_ann import (
    ErenoANN,
    KNOWN_LABELS,
    apply_global_pruning,
    parse_pruning_ratios,
    run_ga_feature_selection,
    sparsity_stats,
)


class FocalLoss(nn.Module):
    """Cross-entropy modulated by (1-pt)^gamma for hard examples (optional class weights)."""

    def __init__(self, *, weight: torch.Tensor | None, gamma: float) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _stratified_quotas(class_counts: Counter, label_order: list[str], k: int) -> dict[int, int]:
    """Per-class-id row budget for reservoir; budgets sum to k (best effort)."""
    pairs = [(i, int(class_counts[lab])) for i, lab in enumerate(label_order) if int(class_counts.get(lab, 0)) > 0]
    if not pairs:
        raise ValueError("No labeled rows in training class counts.")
    ids = [p[0] for p in pairs]
    ns = np.array([p[1] for p in pairs], dtype=np.float64)
    c = len(ids)
    n_total = float(ns.sum())
    raw = k * ns / n_total
    q = np.floor(raw).astype(np.int64)
    if k >= c:
        q = np.maximum(q, 1)
    q = np.minimum(q, ns.astype(np.int64))
    deficit = int(k - int(q.sum()))
    if deficit > 0:
        slack = (ns - q).astype(np.int64)
        rem = raw - q
        order = np.argsort(-rem)
        t = 0
        while deficit > 0 and int(slack.sum()) > 0:
            j = int(order[t % len(order)])
            if slack[j] > 0:
                q[j] += 1
                slack[j] -= 1
                deficit -= 1
            t += 1
            if t > 10 * max(k, 1):
                break
    while deficit > 0:
        slack = (ns - q).astype(np.int64)
        if int(slack.sum()) == 0:
            break
        j = int(np.argmax(slack))
        if slack[j] > 0:
            q[j] += 1
            deficit -= 1
        else:
            break
    surplus = int(q.sum()) - k
    floor_min = 1 if k >= c else 0
    while surplus > 0:
        j = int(np.argmax(q))
        if q[j] > floor_min:
            q[j] -= 1
            surplus -= 1
        else:
            break
    return {ids[i]: int(q[i]) for i in range(c)}


@dataclass
class StreamEncoderState:
    target_col: str
    """Column used for labels (may be 'delay' if 'class' was empty, matching train_ereno_ann)."""
    resolved_target: str
    feature_cols: list[str]
    numeric_cols: list[str]
    obj_cols: list[str]
    dropped_numeric: list[str]
    mean: np.ndarray  # (n_num,)
    std: np.ndarray  # (n_num,)
    cat_maps: dict[str, dict[str, int]]
    labels: list[str]
    label_to_id: dict[str, int]
    num_classes: int
    binary_mode: bool = False
    """When True, map normal->0 and every other multiclass name->1 at encode time."""
    multiclass_label_to_id: dict[str, int] | None = None
    """Original multiclass mapping from pass1 (binary mode only), for diagnostics."""


def _resolve_label_column(first: pd.DataFrame, target_col: str) -> str:
    if target_col in first.columns and first[target_col].notna().any():
        return target_col
    if "delay" in first.columns:
        cand = first["delay"].astype(str).str.strip()
        if bool(cand.isin(KNOWN_LABELS).any()):
            print("[stream] Using 'delay' as label column (target column empty).", flush=True)
            return "delay"
    return target_col


def _pass1_train_stream(
    train_path: Path,
    target_col: str,
    chunk_rows: int,
    variance_threshold: float,
    *,
    binary_mode: bool = False,
) -> tuple[StreamEncoderState, int, Counter]:
    """One full pass: Welford stats, category uniques, class counts; decide variance drops."""
    reader = pd.read_csv(train_path, chunksize=chunk_rows)
    first = next(reader)
    if target_col not in first.columns:
        raise ValueError(f"Missing {target_col!r} in train CSV.")
    resolved = _resolve_label_column(first, target_col)
    if resolved not in first.columns:
        raise ValueError(f"Resolved label column {resolved!r} missing from train CSV.")
    feature_cols = [c for c in first.columns if c != resolved]
    numeric_cols = first[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in feature_cols if c not in numeric_cols]

    n_num = len(numeric_cols)
    count = 0
    mean = np.zeros(n_num, dtype=np.float64)
    M2 = np.zeros(n_num, dtype=np.float64)
    cat_uniques: dict[str, set[str]] = {c: set() for c in obj_cols}
    class_counts: Counter = Counter()

    def update_welford(batch: pd.DataFrame) -> None:
        nonlocal count, mean, M2
        if not numeric_cols:
            return
        x = batch[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        for row in x:
            count += 1
            delta = row - mean
            mean += delta / count
            delta2 = row - mean
            M2 += delta * delta2

    def update_cats(batch: pd.DataFrame) -> None:
        for c in obj_cols:
            cat_uniques[c].update(batch[c].astype(str).fillna("__NA__").unique().tolist())

    def update_classes(batch: pd.DataFrame) -> None:
        class_counts.update(batch[resolved].astype(str).str.strip().tolist())

    def consume(batch: pd.DataFrame) -> None:
        update_welford(batch)
        update_cats(batch)
        update_classes(batch)

    consume(first)
    for chunk in reader:
        consume(chunk)

    if count < 2:
        var = np.zeros(n_num, dtype=np.float64)
    else:
        var = M2 / (count - 1)
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)

    dropped_numeric = [numeric_cols[i] for i in range(n_num) if var[i] <= variance_threshold]
    keep_numeric = [c for c in numeric_cols if c not in set(dropped_numeric)]

    cat_maps: dict[str, dict[str, int]] = {}
    for c in obj_cols:
        vals = sorted(cat_uniques[c])
        cat_maps[c] = {v: i for i, v in enumerate(vals)}

    multiclass_labels = sorted(class_counts.keys())
    multiclass_label_to_id = {lab: i for i, lab in enumerate(multiclass_labels)}
    if binary_mode:
        labels, label_to_id, num_classes = stream_binary_state_fields()
    else:
        labels = multiclass_labels
        label_to_id = multiclass_label_to_id
        num_classes = len(labels)
    std = np.where(var > 1e-18, np.sqrt(var), 1.0)
    # align std to kept numeric indices only
    idx_keep = [numeric_cols.index(c) for c in keep_numeric]
    mean_k = mean[idx_keep] if keep_numeric else np.zeros(0, dtype=np.float64)
    std_k = std[idx_keep] if keep_numeric else np.ones(0, dtype=np.float64)
    std_k = np.where(std_k < 1e-12, 1.0, std_k)

    state = StreamEncoderState(
        target_col=target_col,
        resolved_target=resolved,
        feature_cols=feature_cols,
        numeric_cols=keep_numeric,
        obj_cols=obj_cols,
        dropped_numeric=dropped_numeric,
        mean=mean_k.astype(np.float32),
        std=std_k.astype(np.float32),
        cat_maps=cat_maps,
        labels=labels,
        label_to_id=label_to_id,
        num_classes=num_classes,
        binary_mode=binary_mode,
        multiclass_label_to_id=multiclass_label_to_id if binary_mode else None,
    )
    return state, count, class_counts


def _encode_chunk(
    chunk: pd.DataFrame,
    state: StreamEncoderState,
) -> tuple[np.ndarray, np.ndarray]:
    """Float32 [n, n_features] and int64 labels; drops rows with unknown class."""
    y_raw = chunk[state.resolved_target].astype(str).str.strip().to_numpy()
    if state.binary_mode:
        mask_known = np.array([len(s) > 0 for s in y_raw], dtype=bool)
    else:
        mask_known = np.array([s in state.label_to_id for s in y_raw], dtype=bool)
    chunk = chunk.loc[mask_known].reset_index(drop=True)
    if len(chunk) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    y_str = chunk[state.resolved_target].astype(str).str.strip().to_numpy()
    if state.binary_mode:
        y = np.array([multiclass_label_to_binary_id(s) for s in y_str], dtype=np.int64)
    else:
        y = np.array([state.label_to_id[s] for s in y_str], dtype=np.int64)

    if state.numeric_cols:
        tr_num = chunk[state.numeric_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        tr_num = tr_num.fillna(pd.Series(state.mean, index=state.numeric_cols))
        tr_num = ((tr_num - state.mean) / state.std).to_numpy(dtype=np.float32)
    else:
        tr_num = np.empty((len(chunk), 0), dtype=np.float32)

    if state.obj_cols:
        blocks = []
        for c in state.obj_cols:
            m = state.cat_maps[c]
            s = chunk[c].astype(str).fillna("__NA__")
            codes = s.map(lambda v, mm=m: mm.get(v, -1)).to_numpy(dtype=np.int64)
            blocks.append(codes.astype(np.float32).reshape(-1, 1))
        tr_cat = np.concatenate(blocks, axis=1).astype(np.float32)
    else:
        tr_cat = np.empty((len(chunk), 0), dtype=np.float32)

    x = np.concatenate([tr_num, tr_cat], axis=1).astype(np.float32)
    return x, y


def _feature_names(state: StreamEncoderState) -> list[str]:
    return state.numeric_cols + [f"{c}__cat_code" for c in state.obj_cols]


def _reservoir_uniform_stream(
    train_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform global reservoir of up to k encoded rows (legacy behavior)."""
    pool_x: list[np.ndarray] = []
    pool_y: list[np.ndarray] = []
    n_seen = 0
    reader = pd.read_csv(train_path, chunksize=chunk_rows)
    for chunk in reader:
        x, y = _encode_chunk(chunk, state)
        if len(x) == 0:
            continue
        for i in range(len(x)):
            n_seen += 1
            row_x, row_y = x[i], y[i]
            if len(pool_x) < k:
                pool_x.append(row_x.copy())
                pool_y.append(row_y)
            else:
                j = int(rng.integers(1, n_seen + 1))
                if j <= k:
                    slot = int(rng.integers(0, k))
                    pool_x[slot] = row_x.copy()
                    pool_y[slot] = row_y
    if not pool_x:
        raise ValueError("Reservoir sample empty (no train rows?).")
    return np.stack(pool_x, axis=0), np.asarray(pool_y, dtype=np.int64)


def _reservoir_stratified_stream(
    train_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
    quotas: dict[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One reservoir per class id so GA sees all attack types proportionally (sum sizes = total k)."""
    caps = {c: int(quotas[c]) for c in quotas if int(quotas[c]) > 0}
    pools: dict[int, list[np.ndarray]] = {c: [] for c in caps}
    seen: dict[int, int] = {c: 0 for c in caps}
    reader = pd.read_csv(train_path, chunksize=chunk_rows)
    for chunk in reader:
        x, y = _encode_chunk(chunk, state)
        if len(x) == 0:
            continue
        for i in range(len(x)):
            row_x, cls = x[i], int(y[i])
            if cls not in caps:
                continue
            seen[cls] += 1
            cap = caps[cls]
            pool = pools[cls]
            if len(pool) < cap:
                pool.append(row_x.copy())
            else:
                j = int(rng.integers(1, seen[cls] + 1))
                if j <= cap:
                    slot = int(rng.integers(0, cap))
                    pool[slot] = row_x.copy()
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cls in sorted(caps.keys()):
        for row in pools[cls]:
            xs.append(row)
            ys.append(cls)
    if not xs:
        raise ValueError("Stratified reservoir empty.")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def _train_one_epoch_streaming(
    model: nn.Module,
    train_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: np.random.Generator,
    feature_mask: np.ndarray | None,
) -> float:
    model.train()
    running = 0.0
    batches = 0
    reader = pd.read_csv(train_path, chunksize=chunk_rows)
    for chunk in reader:
        x_np, y_np = _encode_chunk(chunk, state)
        if len(x_np) == 0:
            continue
        if feature_mask is not None:
            x_np = x_np[:, feature_mask]
        perm = rng.permutation(len(x_np))
        x_np = x_np[perm]
        y_np = y_np[perm]
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_np, dtype=torch.long, device=device)
        bs = min(4096, max(256, len(x_t)))
        for i in range(0, len(x_t), bs):
            xb = x_t[i : i + bs]
            yb = y_t[i : i + bs]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
    return running / max(1, batches)


def _eval_streaming(
    model: nn.Module,
    test_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
    device: torch.device,
    feature_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred, _ = _eval_streaming_scored(
        model, test_path, state, chunk_rows, device, feature_mask, return_logits=False
    )
    return y_true, y_pred


def _eval_streaming_scored(
    model: nn.Module,
    test_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
    device: torch.device,
    feature_mask: np.ndarray | None,
    *,
    return_logits: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunked test eval: y_true, y_pred, attack probability (or zeros if return_logits=False)."""
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    reader = pd.read_csv(test_path, chunksize=chunk_rows)
    with torch.no_grad():
        for chunk in reader:
            x_np, y_np = _encode_chunk(chunk, state)
            if len(x_np) == 0:
                continue
            if feature_mask is not None:
                x_np = x_np[:, feature_mask]
            x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
            logits = model(x_t).cpu().numpy()
            pred = logits.argmax(axis=1)
            ys.append(y_np)
            ps.append(pred)
            if return_logits:
                scores.append(attack_scores_from_logits(logits))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_score = np.concatenate(scores) if scores else np.zeros(len(y_true), dtype=np.float64)
    return y_true, y_pred, y_score


def _eval_arrays_scored(
    model: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    device: torch.device,
    feature_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a fixed encoded matrix (e.g. test reservoir for per-epoch eval)."""
    model.eval()
    x = x_np[:, feature_mask] if feature_mask is not None else x_np
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_t).cpu().numpy()
    pred = logits.argmax(axis=1)
    scores = attack_scores_from_logits(logits)
    return y_np, pred, scores


def _metrics_and_cm_from_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    binary_mode: bool,
    classes_idx: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    if binary_mode:
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    else:
        metrics = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=classes_idx)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=classes_idx)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
        cm = confusion_matrix(y_true, y_pred, labels=classes_idx)
    return metrics, cm


def _run_epoch_eval(
    model: nn.Module,
    *,
    epoch: int,
    total_epochs: int,
    eval_scope: str,
    binary_mode: bool,
    classes_idx: np.ndarray,
    labels: list[str],
    mean_batch_loss: float,
    device: torch.device,
    feature_mask: np.ndarray | None,
    epoch_eval_sample: tuple[np.ndarray, np.ndarray] | None,
    test_path: Path | None,
    state: StreamEncoderState,
    chunk_rows: int,
    save_cm_plot: Path | None,
) -> dict[str, Any]:
    if epoch_eval_sample is not None:
        y_true, y_pred, y_score = _eval_arrays_scored(
            model, epoch_eval_sample[0], epoch_eval_sample[1], device, feature_mask
        )
    elif test_path is not None:
        y_true, y_pred, y_score = _eval_streaming_scored(
            model, test_path, state, chunk_rows, device, feature_mask
        )
    else:
        raise ValueError("epoch eval requires epoch_eval_sample or test_path")
    metrics, cm = _metrics_and_cm_from_eval(
        y_true, y_pred, y_score, binary_mode=binary_mode, classes_idx=classes_idx
    )
    print_epoch_eval_report(
        epoch=epoch,
        total_epochs=total_epochs,
        eval_scope=eval_scope,
        labels=labels,
        cm=cm,
        metrics=metrics,
        binary_mode=binary_mode,
        mean_batch_loss=mean_batch_loss,
    )
    if save_cm_plot is not None:
        plot_confusion_matrix(
            cm,
            save_cm_plot,
            title=f"Epoch {epoch}/{total_epochs} — {eval_scope}",
        )
    row: dict[str, Any] = {
        "epoch": int(epoch),
        "mean_batch_loss": float(mean_batch_loss),
        "eval_scope": eval_scope,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(labels),
    }
    if save_cm_plot is not None:
        row["confusion_matrix_plot"] = str(save_cm_plot)
    return row


def _sample_embeddings_stream(
    model: nn.Module,
    test_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
    device: torch.device,
    feature_mask: np.ndarray | None,
    *,
    rows_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified penultimate-layer embeddings for t-SNE (binary caps per class id)."""
    rng = np.random.default_rng(seed)
    caps = {i: int(rows_per_class) for i in range(state.num_classes)}
    pools_x: dict[int, list[np.ndarray]] = {i: [] for i in caps}
    seen: dict[int, int] = {i: 0 for i in caps}
    model.eval()
    reader = pd.read_csv(test_path, chunksize=chunk_rows)
    with torch.no_grad():
        for chunk in reader:
            x_np, y_np = _encode_chunk(chunk, state)
            if len(x_np) == 0:
                continue
            if feature_mask is not None:
                x_np = x_np[:, feature_mask]
            x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
            emb = model.features(x_t).cpu().numpy()
            for cid in np.unique(y_np):
                cid_int = int(cid)
                if cid_int not in caps:
                    continue
                mask = y_np == cid_int
                sub = emb[mask]
                seen[cid_int] += int(mask.sum())
                need = caps[cid_int] - len(pools_x[cid_int])
                if need <= 0:
                    continue
                if len(sub) > need:
                    pick = rng.choice(len(sub), size=need, replace=False)
                    sub = sub[pick]
                pools_x[cid_int].extend(list(sub))
            if all(len(pools_x[i]) >= caps[i] for i in caps):
                break
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cid in sorted(caps.keys()):
        for row in pools_x[cid]:
            xs.append(row)
            ys.append(cid)
    if not xs:
        raise ValueError("No embedding rows sampled from test CSV.")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def _multiclass_labels_aligned_eval(
    test_path: Path,
    state: StreamEncoderState,
    chunk_rows: int,
) -> np.ndarray:
    """Multiclass string labels only for rows kept by _encode_chunk (same chunk order)."""
    out: list[str] = []
    reader = pd.read_csv(test_path, chunksize=chunk_rows)
    for chunk in reader:
        y_raw = chunk[state.resolved_target].astype(str).str.strip().to_numpy()
        if state.binary_mode:
            mask = np.array([len(s) > 0 for s in y_raw], dtype=bool)
        else:
            mask = np.array([s in state.label_to_id for s in y_raw], dtype=bool)
        out.extend(y_raw[mask].tolist())
    return np.asarray(out, dtype=object)


def _write_binary_artifacts(
    *,
    artifacts_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    metrics: dict[str, Any],
    state: StreamEncoderState,
    test_path: Path,
    chunk_rows: int,
    model: nn.Module,
    device: torch.device,
    feature_mask: np.ndarray | None,
    seed: int,
    ga_summary: dict[str, Any],
    feature_name_list: list[str],
    tsne_rows_per_class: int,
    skip_tsne: bool,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(cm, artifacts_dir / "confusion_matrix.png", title="Binary confusion matrix (test)")
    if metrics.get("roc_auc") is not None:
        plot_roc_curve(y_true, y_score, artifacts_dir / "roc_curve.png", title="ROC — attack vs normal")
        plot_pr_curve(y_true, y_score, artifacts_dir / "pr_curve.png", title="Precision–recall — attack vs normal")

    mc_diag: dict[str, Any] = {}
    if state.binary_mode:
        y_mc = _multiclass_labels_aligned_eval(test_path, state, chunk_rows)
        if len(y_mc) == len(y_true):
            mc_diag = multiclass_breakdown(y_true, y_pred, y_mc)
        else:
            mc_diag = {"error": f"multiclass length {len(y_mc)} != eval length {len(y_true)}"}
        (artifacts_dir / "multiclass_diagnostics.json").write_text(
            json.dumps(mc_diag, indent=2), encoding="utf-8"
        )

    if not skip_tsne:
        print("[stream] t-SNE embedding sample from test set...", flush=True)
        emb_x, emb_y = _sample_embeddings_stream(
            model,
            test_path,
            state,
            chunk_rows,
            device,
            feature_mask,
            rows_per_class=tsne_rows_per_class,
            seed=seed,
        )
        plot_tsne_binary(
            emb_x,
            emb_y,
            artifacts_dir / "tsne_binary.png",
            title="t-SNE penultimate activations (normal vs attack)",
            seed=seed,
        )
    save_selected_features(artifacts_dir / "selected_features.json", ga_summary, feature_name_list)
    return mc_diag


def _build_ga_args(
    ns: argparse.Namespace,
    *,
    rare_class_ids: list[int] | None = None,
    rare_bonus: float = 0.0,
    protected_indices: list[int] | None = None,
) -> argparse.Namespace:
    """Minimal namespace for run_ga_feature_selection."""
    return argparse.Namespace(
        seed=ns.seed,
        ga_population=ns.ga_population,
        ga_generations=ns.ga_generations,
        ga_mutation_rate=ns.ga_mutation_rate,
        ga_crossover_rate=ns.ga_crossover_rate,
        ga_min_features=ns.ga_min_features,
        ga_eval_epochs=ns.ga_eval_epochs,
        ga_sample_size=ns.ga_sample_size,
        batch_size=ns.batch_size,
        lr=ns.lr,
        ga_rare_class_ids=list(rare_class_ids or []),
        ga_rare_bonus=float(rare_bonus),
        ga_protected_indices=list(protected_indices or []),
    )


def _parse_hidden_dims(spec: str) -> list[int]:
    out: list[int] = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise ValueError(f"hidden_dims must be positive ints, got '{tok}'.")
        out.append(v)
    if not out:
        raise ValueError("hidden_dims must contain at least one layer width.")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream full ERENO CSVs: GA + train + prune with bounded RAM.")
    p.add_argument("--train-csv", type=Path, default=STREAM_TRAIN_CSV)
    p.add_argument("--test-csv", type=Path, default=STREAM_TEST_CSV)
    p.add_argument("--target-col", type=str, default="class")
    p.add_argument("--chunk-rows", type=int, default=200_000, help="pandas.read_csv chunksize.")
    p.add_argument("--reservoir-rows", type=int, default=120_000, help="Rows for GA reservoir (encoded).")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variance-threshold", type=float, default=0.0)
    p.add_argument("--no-ga", action="store_true")
    p.add_argument("--ga-population", type=int, default=16)
    p.add_argument("--ga-generations", type=int, default=8)
    p.add_argument("--ga-mutation-rate", type=float, default=0.05)
    p.add_argument("--ga-crossover-rate", type=float, default=0.8)
    p.add_argument("--ga-min-features", type=int, default=4)
    p.add_argument("--ga-eval-epochs", type=int, default=3)
    p.add_argument("--ga-sample-size", type=int, default=15_000, help="Cap inside GA on reservoir (0=all).")
    p.add_argument(
        "--pruning-ratios",
        type=str,
        default="0.2,0.4,0.6,0.8,0.9",
        help="Comma-separated ratios in [0,1). Empty to skip.",
    )
    p.add_argument("--output-json", type=Path, default=BINARY_BASELINE_JSON)
    p.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Team-readable markdown (default: <output-json-stem>_eval.md next to JSON).",
    )
    p.add_argument("--save-model", action="store_true", help="Save dense (unpruned) checkpoint after training.")
    p.add_argument(
        "--save-best-pruned",
        action="store_true",
        help="Also save checkpoint at the best pruning ratio from the sweep (by attack F1 if binary).",
    )
    p.add_argument(
        "--epoch-eval",
        choices=("off", "sample", "full"),
        default="sample",
        help="After each epoch: print confusion matrix. sample=stratified test subset (default); full=entire test CSV (slow).",
    )
    p.add_argument(
        "--epoch-eval-rows",
        type=int,
        default=60_000,
        help="Rows in test reservoir when --epoch-eval sample (stratified normal/attack).",
    )
    p.add_argument(
        "--epoch-eval-plots",
        action="store_true",
        help="Save per-epoch confusion matrix PNG under <artifacts-dir>/epochs/ (needs --artifacts-dir or binary default).",
    )
    p.add_argument(
        "--uniform-ga-reservoir",
        action="store_true",
        help="One global uniform reservoir for GA (legacy). Default is class-stratified proportional to train counts.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal loss exponent; 0 keeps weighted cross-entropy.",
    )
    p.add_argument(
        "--class-weights",
        type=str,
        choices=["balanced", "sqrt", "off"],
        default=None,
        help="Loss class-weight scheme. 'balanced' (1/n_c), 'sqrt' (1/sqrt(n_c)), 'off' (uniform).",
    )
    p.add_argument(
        "--ga-rare-classes",
        type=str,
        default="",
        help="Comma-separated class names to upweight in GA fitness (e.g. 'masquerade_fake_fault,masquerade_fake_normal').",
    )
    p.add_argument(
        "--ga-rare-bonus",
        type=float,
        default=0.0,
        help="Multiplier on SUM of recalls over --ga-rare-classes added to GA fitness.",
    )
    p.add_argument(
        "--ga-protect-features",
        type=str,
        default="",
        help="Comma-separated feature names that GA must keep on. Use names from `_feature_names` (e.g. 'stDiff,timestampDiff,SqNum,gooseAppid__cat_code').",
    )
    p.add_argument(
        "--hidden-dims",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer widths, e.g. '128,64' (default), '256,128' or '256,128,64'.",
    )
    p.add_argument(
        "--binary-mode",
        action="store_true",
        help="Binary IDS: normal=0, every other multiclass label=1. Multiclass remains the default.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for plots, selected_features.json, and extras. "
        "Default: sibling folder of --output-json named '<stem>_artifacts'.",
    )
    p.add_argument(
        "--tsne-rows-per-class",
        type=int,
        default=4000,
        help="Max test rows per class for t-SNE plot (when binary artifacts are written).",
    )
    p.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip t-SNE / curve plots (metrics still computed).",
    )
    return p.parse_args()


def _print_binary_distribution(
    *,
    train_counts: Counter,
    test_counts: Counter,
    n_train_rows: int,
    n_test_rows: int,
) -> None:
    train_b = binary_counts_from_multiclass(train_counts)
    test_b = binary_counts_from_multiclass(test_counts)
    print(f"[binary] train rows={n_train_rows} {format_binary_count_line(train_b)}", flush=True)
    print(f"[binary] test  rows={n_test_rows} {format_binary_count_line(test_b)}", flush=True)
    tr_n = train_b[NORMAL_CLASS_NAME]
    tr_a = train_b[ATTACK_CLASS_NAME]
    te_n = test_b[NORMAL_CLASS_NAME]
    te_a = test_b[ATTACK_CLASS_NAME]
    if tr_n + tr_a > 0 and te_n + te_a > 0:
        print(
            f"[binary] rates train: normal={tr_n / (tr_n + tr_a):.4f} attack={tr_a / (tr_n + tr_a):.4f} | "
            f"test: normal={te_n / (te_n + te_a):.4f} attack={te_a / (te_n + te_a):.4f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert_comma_separated_csv(args.train_csv, role="Train")
    assert_comma_separated_csv(args.test_csv, role="Test")

    print("[stream] pass1: full train scan (stats + class histogram)...", flush=True)
    state, n_train_rows, class_counts = _pass1_train_stream(
        args.train_csv,
        args.target_col,
        args.chunk_rows,
        args.variance_threshold,
        binary_mode=bool(args.binary_mode),
    )
    print(f"[stream] train_rows={n_train_rows} labels={state.num_classes} dropped_numeric={len(state.dropped_numeric)}", flush=True)
    if args.binary_mode:
        train_b = binary_counts_from_multiclass(class_counts)
        print(f"[binary] train class counts: {format_binary_count_line(train_b)}", flush=True)
        print("[stream] scanning test CSV for binary label distribution...", flush=True)
        test_mc, n_test_rows = count_binary_labels_in_csv(
            args.test_csv,
            target_col=args.target_col,
            chunk_rows=args.chunk_rows,
            resolve_label_column=_resolve_label_column,
        )
        _print_binary_distribution(
            train_counts=class_counts,
            test_counts=test_mc,
            n_train_rows=n_train_rows,
            n_test_rows=n_test_rows,
        )

    if args.binary_mode and float(args.ga_rare_bonus) > 0.0 and not (args.ga_rare_classes or "").strip():
        args.ga_rare_classes = ATTACK_CLASS_NAME
        print(f"[stream] binary GA rare bonus: default --ga-rare-classes {ATTACK_CLASS_NAME!r}", flush=True)
    rare_class_names = [s.strip() for s in (args.ga_rare_classes or "").split(",") if s.strip()]
    rare_class_ids: list[int] = []
    rare_unknown: list[str] = []
    for name in rare_class_names:
        if args.binary_mode:
            if name in state.label_to_id:
                rare_class_ids.append(int(state.label_to_id[name]))
            elif state.multiclass_label_to_id and name in state.multiclass_label_to_id:
                rare_class_ids.append(multiclass_label_to_binary_id(name))
            else:
                rare_unknown.append(name)
        elif name in state.label_to_id:
            rare_class_ids.append(int(state.label_to_id[name]))
        else:
            rare_unknown.append(name)
    if rare_unknown:
        print(f"[stream] WARN: --ga-rare-classes ignored unknown labels: {rare_unknown}", flush=True)
    if rare_class_ids and args.ga_rare_bonus > 0.0:
        print(
            f"[stream] GA rare-class bonus {args.ga_rare_bonus} on {[state.labels[i] for i in rare_class_ids]}",
            flush=True,
        )

    feature_name_list = _feature_names(state)
    feature_name_to_idx = {n: i for i, n in enumerate(feature_name_list)}
    protect_names = [s.strip() for s in (args.ga_protect_features or "").split(",") if s.strip()]
    protected_indices: list[int] = []
    protect_unknown: list[str] = []
    for name in protect_names:
        if name in feature_name_to_idx:
            protected_indices.append(feature_name_to_idx[name])
        else:
            protect_unknown.append(name)
    if protect_unknown:
        print(f"[stream] WARN: --ga-protect-features ignored unknown features: {protect_unknown}", flush=True)
    if protected_indices:
        print(
            f"[stream] GA protected features ({len(protected_indices)}): "
            f"{[feature_name_list[i] for i in protected_indices]}",
            flush=True,
        )

    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    ga_mask: np.ndarray | None = None
    ga_summary: dict[str, Any] = {"enabled": False}
    ga_reservoir_mode = "disabled"
    ga_quotas: dict[str, int] | None = None
    if not args.no_ga:
        ga_reservoir_mode = "uniform" if args.uniform_ga_reservoir else "stratified"
        print(
            f"[stream] reservoir sample k={args.reservoir_rows} mode={ga_reservoir_mode} for GA...",
            flush=True,
        )
        if args.uniform_ga_reservoir:
            xr, yr = _reservoir_uniform_stream(
                args.train_csv, state, args.chunk_rows, args.reservoir_rows, rng
            )
        else:
            ga_count_source = (
                Counter(binary_counts_from_multiclass(class_counts))
                if args.binary_mode
                else class_counts
            )
            quotas = _stratified_quotas(ga_count_source, state.labels, args.reservoir_rows)
            ga_quotas = {state.labels[cid]: int(n) for cid, n in sorted(quotas.items())}
            xr, yr = _reservoir_stratified_stream(
                args.train_csv, state, args.chunk_rows, quotas, rng
            )
        ga_args = _build_ga_args(
            args,
            rare_class_ids=rare_class_ids,
            rare_bonus=args.ga_rare_bonus,
            protected_indices=protected_indices,
        )
        ga_mask, ga_summary = run_ga_feature_selection(xr, yr, feature_name_list, ga_args, device)
        ga_summary["rare_class_ids"] = rare_class_ids
        ga_summary["rare_class_names"] = [state.labels[i] for i in rare_class_ids]
        ga_summary["rare_bonus"] = float(args.ga_rare_bonus)
        print(f"[stream] GA selected {int(ga_mask.sum())} / {len(ga_mask)} features", flush=True)
    else:
        ga_mask = np.ones(len(state.numeric_cols) + len(state.obj_cols), dtype=bool)

    n_feat = int(ga_mask.sum())
    model = ErenoANN(input_dim=n_feat, num_classes=state.num_classes, hidden_dims=hidden_dims).to(device)
    print(f"[stream] ANN hidden_dims={hidden_dims} input_dim={n_feat}", flush=True)
    classes_idx = np.arange(state.num_classes)
    weight_counts = (
        binary_counts_from_multiclass(class_counts) if state.binary_mode else dict(class_counts)
    )
    parts_w: list[np.ndarray] = []
    for i, lab in enumerate(state.labels):
        cnt = int(weight_counts.get(lab, 0))
        if cnt > 0:
            parts_w.append(np.full(cnt, i, dtype=np.int64))
    y_dummy = np.concatenate(parts_w) if parts_w else np.array([0], dtype=np.int64)
    cw_scheme = args.class_weights if args.class_weights is not None else "balanced"
    if cw_scheme == "balanced":
        weights = compute_class_weight(class_weight="balanced", classes=classes_idx, y=y_dummy)
        class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    elif cw_scheme == "sqrt":
        cnts = np.array(
            [max(1, int(weight_counts.get(state.labels[i], 0))) for i in classes_idx],
            dtype=np.float64,
        )
        inv = 1.0 / np.sqrt(cnts)
        weights = inv * (len(cnts) / inv.sum())
        class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights_t = None
    focal_gamma = float(args.focal_gamma)
    if focal_gamma > 0.0:
        criterion = FocalLoss(weight=class_weights_t, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    print(
        f"[stream] criterion={'focal' if focal_gamma > 0 else 'cross_entropy'} "
        f"gamma={focal_gamma} class_weights={cw_scheme}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_eval_mode = str(args.epoch_eval)
    epoch_eval_sample: tuple[np.ndarray, np.ndarray] | None = None
    epoch_eval_scope = "off"
    if epoch_eval_mode == "sample":
        k_eval = max(1000, int(args.epoch_eval_rows))
        print(
            f"[stream] building stratified test sample (n={k_eval}) for per-epoch confusion matrices...",
            flush=True,
        )
        if args.binary_mode:
            test_mc_ep, _ = count_binary_labels_in_csv(
                args.test_csv,
                target_col=args.target_col,
                chunk_rows=args.chunk_rows,
                resolve_label_column=_resolve_label_column,
            )
            test_b = binary_counts_from_multiclass(test_mc_ep)
            test_counter = Counter(
                {NORMAL_CLASS_NAME: test_b[NORMAL_CLASS_NAME], ATTACK_CLASS_NAME: test_b[ATTACK_CLASS_NAME]}
            )
            quotas = _stratified_quotas(test_counter, state.labels, k_eval)
            epoch_eval_sample = _reservoir_stratified_stream(
                args.test_csv, state, args.chunk_rows, quotas, rng
            )
        else:
            epoch_eval_sample = _reservoir_uniform_stream(
                args.test_csv, state, args.chunk_rows, k_eval, rng
            )
        epoch_eval_scope = f"test-sample (n={len(epoch_eval_sample[0])})"
    elif epoch_eval_mode == "full":
        epoch_eval_scope = "test-full"
        print(
            "[stream] per-epoch eval uses the full test CSV (slow on multi-million-row files).",
            flush=True,
        )

    epoch_plots_dir: Path | None = None
    if args.epoch_eval_plots:
        base_art = args.artifacts_dir
        if base_art is None and args.binary_mode:
            base_art = args.output_json.parent / f"{args.output_json.stem}_artifacts"
        if base_art is not None:
            epoch_plots_dir = Path(base_art) / "epochs"
            epoch_plots_dir.mkdir(parents=True, exist_ok=True)

    print("[stream] training (each epoch = one full pass over train CSV)...", flush=True)
    losses: list[float] = []
    epoch_eval_history: list[dict[str, Any]] = []
    for ep in range(args.epochs):
        le = _train_one_epoch_streaming(
            model, args.train_csv, state, args.chunk_rows, criterion, optimizer, device, rng, ga_mask
        )
        losses.append(le)
        ep_no = ep + 1
        if epoch_eval_mode != "off":
            cm_plot = (
                (epoch_plots_dir / f"epoch_{ep_no:02d}_confusion_matrix.png")
                if epoch_plots_dir is not None
                else None
            )
            hist_row = _run_epoch_eval(
                model,
                epoch=ep_no,
                total_epochs=args.epochs,
                eval_scope=epoch_eval_scope,
                binary_mode=bool(args.binary_mode),
                classes_idx=classes_idx,
                labels=list(state.labels),
                mean_batch_loss=le,
                device=device,
                feature_mask=ga_mask,
                epoch_eval_sample=epoch_eval_sample,
                test_path=args.test_csv if epoch_eval_mode == "full" else None,
                state=state,
                chunk_rows=args.chunk_rows,
                save_cm_plot=cm_plot,
            )
            epoch_eval_history.append(hist_row)
        else:
            print(f"[stream] epoch {ep_no:02d}/{args.epochs} mean_batch_loss={le:.4f}", flush=True)

    print("[stream] eval: full test CSV (chunked)...", flush=True)
    y_true, y_pred, y_score = _eval_streaming_scored(
        model, args.test_csv, state, args.chunk_rows, device, ga_mask
    )
    if args.binary_mode:
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=classes_idx)
        )
    else:
        metrics = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=classes_idx)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=classes_idx)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
    if args.binary_mode:
        print(f"[stream] baseline (test set, dense model): {format_binary_console_summary(metrics)}", flush=True)
    else:
        print(
            f"[stream] baseline (test set): macro_f1={metrics['macro_f1']:.4f} "
            f"mcc={metrics['mcc']:.4f} accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )

    per_class = classification_report(
        y_true,
        y_pred,
        labels=classes_idx,
        target_names=list(state.labels),
        output_dict=True,
        zero_division=0,
    )
    cm = (
        np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        if args.binary_mode
        else confusion_matrix(y_true, y_pred, labels=classes_idx)
    )
    recalls = [(lab, float(per_class[lab]["recall"])) for lab in state.labels if lab in per_class]
    recalls.sort(key=lambda t: t[1])
    print("[stream] per-class recall (sorted low->high): " + ", ".join(f"{a}={b:.3f}" for a, b in recalls), flush=True)
    print("[stream] confusion_matrix (rows=true, cols=pred):\n" + np.array2string(cm, separator=","), flush=True)

    prune_list = parse_pruning_ratios(args.pruning_ratios or "")
    pruning_results: list[dict[str, Any]] = []
    for ratio in prune_list:
        m = deepcopy(model)
        apply_global_pruning(m, amount=ratio)
        yt, yp, ys = _eval_streaming_scored(m, args.test_csv, state, args.chunk_rows, device, ga_mask)
        if args.binary_mode:
            pm = compute_binary_metrics(yt, yp, ys)
            pm["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
        else:
            pm = {
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0, labels=classes_idx)),
                "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0, labels=classes_idx)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "accuracy": float(accuracy_score(yt, yp)),
                "mcc": float(matthews_corrcoef(yt, yp)),
            }
        sparsity = sparsity_stats(m)
        pruning_results.append(
            {
                "target_pruning_ratio": float(ratio),
                "metrics": pm,
                "sparsity": sparsity,
            }
        )
        pct = int(ratio * 100)
        if args.binary_mode:
            print(
                f"[stream] prune {pct:>2d}%: {format_binary_console_summary(pm)} "
                f"sparsity={sparsity['sparsity_ratio']:.1%}",
                flush=True,
            )
        else:
            print(
                f"[stream] prune {pct:>2d}%: macro_f1={pm['macro_f1']:.4f} mcc={pm['mcc']:.4f} "
                f"sparsity={sparsity['sparsity_ratio']:.1%}",
                flush=True,
            )

    best_pruning = pick_best_pruning_sweep(pruning_results, binary_mode=bool(args.binary_mode))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    train_binary_counts = binary_counts_from_multiclass(class_counts) if args.binary_mode else None
    test_binary_counts = None
    if args.binary_mode:
        test_mc, _ = count_binary_labels_in_csv(
            args.test_csv,
            target_col=args.target_col,
            chunk_rows=args.chunk_rows,
            resolve_label_column=_resolve_label_column,
        )
        test_binary_counts = binary_counts_from_multiclass(test_mc)

    payload = {
        "mode": "stream_full_coverage",
        "task_type": task_type_for_mode(bool(args.binary_mode)),
        "binary_mode": bool(args.binary_mode),
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "chunk_rows": args.chunk_rows,
        "reservoir_rows": args.reservoir_rows,
        "ga_reservoir_mode": ga_reservoir_mode,
        "ga_reservoir_class_rows": ga_quotas,
        "focal_gamma": focal_gamma,
        "class_weights": cw_scheme,
        "ga_rare_classes": [state.labels[i] for i in rare_class_ids],
        "ga_rare_bonus": float(args.ga_rare_bonus),
        "ga_protected_features": [feature_name_list[i] for i in protected_indices],
        "hidden_dims": hidden_dims,
        "train_rows_seen": n_train_rows,
        "train_class_counts": {str(k): int(v) for k, v in sorted(class_counts.items(), key=lambda kv: str(kv[0]))},
        "train_binary_class_counts": train_binary_counts,
        "test_binary_class_counts": test_binary_counts,
        "dropped_numeric_variance": state.dropped_numeric,
        "labels": state.labels,
        "multiclass_labels": sorted(class_counts.keys()) if args.binary_mode else state.labels,
        "ga_feature_selection": ga_summary,
        "training": {
            "epochs": args.epochs,
            "losses": losses,
            "seed": args.seed,
            "device": str(device),
            "epoch_eval": epoch_eval_mode,
            "epoch_eval_rows": int(args.epoch_eval_rows) if epoch_eval_mode == "sample" else None,
        },
        "epoch_eval_history": epoch_eval_history,
        "baseline_metrics": metrics,
        "baseline_per_class": per_class,
        "baseline_confusion_matrix": cm.tolist(),
        "pruning_sweep": pruning_results,
        "best_pruning_sweep": best_pruning,
    }
    if args.binary_mode:
        payload["baseline_attack_scores_summary"] = {
            "mean": float(np.mean(y_score)),
            "std": float(np.std(y_score)),
            "min": float(np.min(y_score)),
            "max": float(np.max(y_score)),
        }

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None and args.binary_mode:
        artifacts_dir = args.output_json.parent / f"{args.output_json.stem}_artifacts"
    mc_diag: dict[str, Any] = {}
    if args.binary_mode and artifacts_dir is not None:
        mc_diag = _write_binary_artifacts(
            artifacts_dir=Path(artifacts_dir),
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            state=state,
            test_path=args.test_csv,
            chunk_rows=args.chunk_rows,
            model=model,
            device=device,
            feature_mask=ga_mask,
            seed=args.seed,
            ga_summary=ga_summary,
            feature_name_list=[n for n, keep in zip(feature_name_list, ga_mask) if keep],
            tsne_rows_per_class=int(args.tsne_rows_per_class),
            skip_tsne=bool(args.skip_viz),
        )
        payload["multiclass_diagnostics"] = mc_diag
        payload["artifacts_dir"] = str(artifacts_dir)

    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[stream] wrote JSON: {args.output_json}", flush=True)

    summary_md = args.summary_md or args.output_json.with_name(f"{args.output_json.stem}_eval.md")
    n_test_rows = int(len(y_true))
    dataset_info = {
        "train_rows": int(n_train_rows),
        "test_rows": n_test_rows,
        "train_binary_class_counts": train_binary_counts,
        "eval_binary_class_counts": test_binary_counts,
    }
    checkpoint_dense: Path | None = None
    checkpoint_pruned: Path | None = None

    if args.save_model:
        checkpoint_dense = args.output_json.with_name(f"{args.output_json.stem}_seed{args.seed}.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "pruning_applied": False,
                "pruning_ratio": 0.0,
                "input_dim": n_feat,
                "num_classes": state.num_classes,
                "labels": state.labels,
                "hidden_dims": hidden_dims,
                "ga_feature_mask": ga_mask.tolist(),
                "feature_names": [n for n, keep in zip(_feature_names(state), ga_mask) if keep],
                "seed": args.seed,
                "encoder": {
                    "numeric_cols": state.numeric_cols,
                    "obj_cols": state.obj_cols,
                    "mean": state.mean.tolist(),
                    "std": state.std.tolist(),
                    "cat_maps": {k: {a: b for a, b in v.items()} for k, v in state.cat_maps.items()},
                    "target_col": state.target_col,
                    "resolved_target": state.resolved_target,
                    "label_to_id": state.label_to_id,
                    "binary_mode": state.binary_mode,
                    "multiclass_label_to_id": state.multiclass_label_to_id,
                },
            },
            checkpoint_dense,
        )
        payload["checkpoint_dense"] = str(checkpoint_dense)
        print(f"[stream] saved dense checkpoint: {checkpoint_dense}", flush=True)

    if args.save_best_pruned and best_pruning is not None:
        ratio = float(best_pruning["target_pruning_ratio"])
        m_pruned = deepcopy(model)
        apply_global_pruning(m_pruned, amount=ratio)
        pct = int(ratio * 100)
        checkpoint_pruned = args.output_json.with_name(
            f"{args.output_json.stem}_pruned{pct}_seed{args.seed}.pt"
        )
        torch.save(
            {
                "state_dict": m_pruned.state_dict(),
                "pruning_applied": True,
                "pruning_ratio": ratio,
                "sparsity": sparsity_stats(m_pruned),
                "input_dim": n_feat,
                "num_classes": state.num_classes,
                "labels": state.labels,
                "hidden_dims": hidden_dims,
                "ga_feature_mask": ga_mask.tolist(),
                "feature_names": [n for n, keep in zip(_feature_names(state), ga_mask) if keep],
                "seed": args.seed,
                "selection_metric": best_pruning.get("selection_metric"),
                "selection_value": best_pruning.get("selection_value"),
                "encoder": {
                    "numeric_cols": state.numeric_cols,
                    "obj_cols": state.obj_cols,
                    "mean": state.mean.tolist(),
                    "std": state.std.tolist(),
                    "cat_maps": {k: {a: b for a, b in v.items()} for k, v in state.cat_maps.items()},
                    "target_col": state.target_col,
                    "resolved_target": state.resolved_target,
                    "label_to_id": state.label_to_id,
                    "binary_mode": state.binary_mode,
                    "multiclass_label_to_id": state.multiclass_label_to_id,
                },
            },
            checkpoint_pruned,
        )
        payload["checkpoint_pruned"] = str(checkpoint_pruned)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[stream] saved best pruned checkpoint ({pct}%): {checkpoint_pruned}", flush=True)

    write_stream_run_summary_md(
        summary_md,
        binary_mode=bool(args.binary_mode),
        metrics=metrics,
        pruning_results=pruning_results,
        best_pruning=best_pruning,
        dataset_info=dataset_info,
        seed=int(args.seed),
        output_json=args.output_json,
        checkpoint_dense=checkpoint_dense,
        checkpoint_pruned=checkpoint_pruned,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir is not None else None,
        train_csv=str(args.train_csv),
        test_csv=str(args.test_csv),
    )
    print(f"[stream] wrote team summary: {summary_md}", flush=True)

    print_stream_run_footer(
        binary_mode=bool(args.binary_mode),
        metrics=metrics,
        best_pruning=best_pruning,
        summary_md=summary_md,
        output_json=args.output_json,
        checkpoint_dense=checkpoint_dense,
        checkpoint_pruned=checkpoint_pruned,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir is not None else None,
    )


if __name__ == "__main__":
    main()
