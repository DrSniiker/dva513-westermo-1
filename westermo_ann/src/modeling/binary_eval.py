"""Binary IDS metrics, curves, and plots for the streaming ERENO trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from modeling.ereno_labels import ATTACK_CLASS_ID, NORMAL_CLASS_ID, NORMAL_CLASS_NAME, multiclass_label_to_binary_id


def attack_scores_from_logits(logits: np.ndarray) -> np.ndarray:
    """P(attack) from softmax logits; shape (n, 2) for binary classifiers."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"Expected binary logits (n, 2+), got {logits.shape}")
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    prob = exp / exp.sum(axis=1, keepdims=True)
    return prob[:, ATTACK_CLASS_ID]


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: int = ATTACK_CLASS_ID,
) -> dict[str, Any]:
    """Full binary metrics with attack as the positive class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    labels = [NORMAL_CLASS_ID, ATTACK_CLASS_ID]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    pos = positive_label
    metrics: dict[str, Any] = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [NORMAL_CLASS_NAME, "attack"],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }

    for name, cid in (("normal", NORMAL_CLASS_ID), ("attack", ATTACK_CLASS_ID)):
        mask_t = y_true == cid
        mask_p = y_pred == cid
        metrics[f"{name}_precision"] = float(precision_score(y_true == cid, y_pred == cid, zero_division=0))
        metrics[f"{name}_recall"] = float(recall_score(y_true == cid, y_pred == cid, zero_division=0))
        metrics[f"{name}_f1"] = float(f1_score(y_true == cid, y_pred == cid, zero_division=0))

    metrics["precision"] = metrics["attack_precision"]
    metrics["recall"] = metrics["attack_recall"]
    metrics["f1"] = metrics["attack_f1"]

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(
            _safe_auc(y_true, y_score, positive_label=pos, curve_fn=roc_curve)
        )
        metrics["pr_auc"] = float(average_precision_score(y_true == pos, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, *, positive_label: int, curve_fn) -> float:
    y_bin = (y_true == positive_label).astype(np.int64)
    fpr, tpr, _ = curve_fn(y_bin, y_score)
    return float(auc(fpr, tpr))


def multiclass_breakdown(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_true_multiclass: np.ndarray,
) -> dict[str, Any]:
    """Per true multiclass label: counts and binary recall (attack detected)."""
    by_class: dict[str, dict[str, Any]] = {}
    for lab in sorted(set(y_true_multiclass.tolist())):
        lab_s = str(lab)
        mask = y_true_multiclass == lab_s
        n = int(mask.sum())
        if n == 0:
            continue
        pred_attack = int((y_pred_binary[mask] == ATTACK_CLASS_ID).sum())
        by_class[lab_s] = {
            "rows": n,
            "predicted_normal": int((y_pred_binary[mask] == NORMAL_CLASS_ID).sum()),
            "predicted_attack": pred_attack,
            "attack_recall": float(pred_attack / n),
            "true_binary_id": int(multiclass_label_to_binary_id(lab_s)),
        }
    return {"per_multiclass_label": by_class}


EDGE_SINGLE_CORE_CPU_BUDGET_PCT = 25.0


def _fmt_pct(x: float | None) -> str:
    return f"{x:.2f}%" if x is not None and np.isfinite(x) else "n/a"


def _fmt_float(x: float | None, *, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if x is not None and np.isfinite(x) else "n/a"


def _edge_cpu_verdict(single_core_pct: float | None, budget_pct: float = EDGE_SINGLE_CORE_CPU_BUDGET_PCT) -> str:
    if single_core_pct is None or not np.isfinite(single_core_pct):
        return "CPU peak unknown (install `psutil` and re-run)."
    if single_core_pct <= budget_pct:
        return f"Within typical edge budget ({budget_pct:.0f}% of one CPU core)."
    return f"Above typical edge budget ({budget_pct:.0f}% of one CPU core) — consider pruning or a smaller model."


def format_binary_console_summary(metrics: dict[str, Any]) -> str:
    """One-line human summary for logs."""
    return (
        f"attack_F1={_fmt_float(metrics.get('f1'))} "
        f"attack_recall={_fmt_float(metrics.get('recall'))} "
        f"false_alarm_rate(FPR)={_fmt_float(metrics.get('false_positive_rate'))} "
        f"missed_attack_rate(FNR)={_fmt_float(metrics.get('false_negative_rate'))} "
        f"MCC={_fmt_float(metrics.get('mcc'))}"
    )


def write_binary_summary_md(
    path: Path,
    metrics: dict[str, Any],
    *,
    inference: dict[str, Any] | None = None,
    model_stats: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    seed: int | None = None,
) -> None:
    """Team-facing markdown: normal vs attack, confusion matrix, edge runtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    tn, fp, fn, tp = int(metrics.get("tn", cm[0][0])), int(metrics.get("fp", cm[0][1])), int(
        metrics.get("fn", cm[1][0])
    ), int(metrics.get("tp", cm[1][1]))

    lines = [
        "# Binary IDS evaluation (normal vs attack)",
        "",
        "The model predicts **normal** traffic (label 0) or **attack** (label 1). "
        "All original attack types (replay, masquerade, etc.) are grouped as **attack**.",
        "",
    ]
    if seed is not None:
        lines.append(f"**Random seed:** {seed}")
        lines.append("")
    if dataset_info:
        tr = dataset_info.get("train_binary_class_counts") or {}
        te = dataset_info.get("eval_binary_class_counts") or {}
        lines.extend(
            [
                "## Dataset (after publish)",
                "",
                f"- Train rows: {dataset_info.get('train_rows', 'n/a')}",
                f"- Test rows: {dataset_info.get('test_rows', 'n/a')}",
                f"- Train — normal: {tr.get('normal', 0)}, attack: {tr.get('attack', 0)}",
                f"- Test — normal: {te.get('normal', 0)}, attack: {te.get('attack', 0)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Detection quality (test set)",
            "",
            "| What it means | Value |",
            "|---|---:|",
            f"| **Attack F1** (main score) | {_fmt_float(metrics.get('f1'))} |",
            f"| Attack recall (catch attacks) | {_fmt_float(metrics.get('recall'))} |",
            f"| Attack precision (alarms that are real) | {_fmt_float(metrics.get('precision'))} |",
            f"| Missed attacks (FNR) | {_fmt_float(metrics.get('false_negative_rate'))} |",
            f"| False alarms on normal (FPR) | {_fmt_float(metrics.get('false_positive_rate'))} |",
            f"| Overall accuracy | {_fmt_float(metrics.get('accuracy'))} |",
            f"| MCC | {_fmt_float(metrics.get('mcc'))} |",
            f"| ROC-AUC (attack score) | {_fmt_float(metrics.get('roc_auc'))} |",
            "",
            "## Confusion matrix",
            "",
            "Rows = true label, columns = model prediction.",
            "",
            "|  | Predicted normal | Predicted attack |",
            "|---|---:|---:|",
            f"| **True normal** | {tn} | {fp} |",
            f"| **True attack** | {fn} | {tp} |",
            "",
            f"- **True negatives (TN):** {tn} normal correctly accepted",
            f"- **False positives (FP):** {fp} normal wrongly flagged as attack",
            f"- **False negatives (FN):** {fn} attacks missed",
            f"- **True positives (TP):** {tp} attacks correctly detected",
            "",
        ]
    )

    breakdown = metrics.get("multiclass_breakdown") or {}
    per_lab = breakdown.get("per_multiclass_label") or {}
    if per_lab:
        lines.extend(
            [
                "## By original attack type (still binary prediction)",
                "",
                "Shows how often each original class was predicted as attack.",
                "",
                "| Original class | Test rows | Predicted attack | Attack recall |",
                "|---|---:|---:|---:|",
            ]
        )
        for lab, row in sorted(per_lab.items()):
            lines.append(
                f"| {lab} | {row['rows']} | {row['predicted_attack']} | {_fmt_float(row.get('attack_recall'))} |"
            )
        lines.append("")

    if inference is not None or model_stats is not None:
        mem = (inference or {}).get("memory") or {}
        ms = (inference or {}).get("canonical_inference_ms_per_sample")
        ms1 = (inference or {}).get("inference_ms_per_sample_batch1")
        rss = mem.get("cpu_rss_mb_peak_during_inference")
        cpu_sc = mem.get("cpu_percent_single_core_peak_during_inference")
        sz = (model_stats or {}).get("estimated_size_mb_fp32")
        params = (model_stats or {}).get("total_params")
        lines.extend(
            [
                "## Edge / runtime (for device fit)",
                "",
                "| Metric | Value | Notes |",
                "|---|---:|---|",
                f"| Model size (FP32 est.) | {_fmt_float(sz, digits=2)} MB | Weights only |",
                f"| Parameters | {params if params is not None else 'n/a'} | |",
                f"| Inference latency | {_fmt_float(ms, digits=4)} ms/sample | Canonical batch benchmark |",
                f"| Inference (batch=1) | {_fmt_float(ms1, digits=4)} ms/sample | Worst-case per packet |",
                f"| CPU RAM peak (process) | {_fmt_float(rss, digits=1)} MB | During timed inference |",
                f"| CPU use (one core peak) | {_fmt_pct(cpu_sc)} | Compare to {EDGE_SINGLE_CORE_CPU_BUDGET_PCT:.0f}% budget |",
                "",
                f"**Edge CPU verdict:** {_edge_cpu_verdict(cpu_sc)}",
                "",
            ]
        )
        if mem.get("notes"):
            lines.append("Notes: " + " ".join(str(n) for n in mem["notes"]))
            lines.append("")

    lines.append(
        "_Full numbers (including GA, pruning, per-seed runs) are in the matching `.json` file in `reports/`._"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_binary_aggregate_summary_md(
    path: Path,
    seeds: list[int],
    aggregate: dict[str, dict[str, float]],
    per_seed_rows: list[dict[str, Any]],
) -> None:
    """Multi-seed team summary (mean ± std)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Binary IDS evaluation — multiple seeds",
        "",
        f"**Seeds:** {', '.join(str(s) for s in seeds)}",
        "",
        "## Average performance (test set)",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("attack_f1", "Attack F1"),
        ("attack_recall", "Attack recall"),
        ("attack_precision", "Attack precision"),
        ("false_negative_rate", "Missed attacks (FNR)"),
        ("false_positive_rate", "False alarms (FPR)"),
        ("mcc", "MCC"),
        ("accuracy", "Accuracy"),
    ]:
        if key in aggregate:
            m = aggregate[key]
            lines.append(f"| {label} | {m['mean']:.4f} | {m['std']:.4f} |")

    lines.extend(["", "## Edge / runtime (mean ± std)", "", "| Metric | Mean | Std |", "|---|---:|---:|"])
    for key, label in [
        ("inference_ms_per_sample", "Inference ms/sample"),
        ("model_size_mb_fp32", "Model size MB (FP32)"),
        ("cpu_rss_peak_mb", "CPU RAM peak MB"),
        ("cpu_percent_single_core_peak", "CPU % one core peak"),
    ]:
        if key not in aggregate:
            continue
        m = aggregate[key]
        lines.append(f"| {label} | {m['mean']:.4f} | {m['std']:.4f} |")

    if "cpu_percent_single_core_peak" in aggregate:
        mean_cpu = aggregate["cpu_percent_single_core_peak"]["mean"]
        lines.extend(
            [
                "",
                f"**Edge CPU verdict (mean):** {_edge_cpu_verdict(mean_cpu)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Per seed",
            "",
            "| Seed | Attack F1 | Recall | FPR | FNR | MCC | ms/sample | Model MB | CPU 1-core % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed_rows:
        met = row.get("metrics") or {}
        ms = row.get("inference_ms_per_sample")
        sz = row.get("model_size_mb_fp32")
        cpu = row.get("cpu_percent_single_core_peak")
        lines.append(
            f"| {row['seed']} | {met.get('f1', 0):.4f} | {met.get('recall', 0):.4f} | "
            f"{met.get('false_positive_rate', 0):.4f} | {met.get('false_negative_rate', 0):.4f} | "
            f"{met.get('mcc', 0):.4f} | "
            f"{_fmt_float(ms)} | {_fmt_float(sz, digits=2)} | {_fmt_pct(cpu)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def primary_score_from_metrics(metrics: dict[str, Any], *, binary_mode: bool) -> tuple[str, float]:
    """Label and value used to rank models (attack F1 when binary)."""
    if binary_mode:
        return "attack_f1", float(metrics.get("f1", metrics.get("attack_f1", 0.0)))
    return "macro_f1", float(metrics.get("macro_f1", 0.0))


def pick_best_pruning_sweep(
    pruning_results: list[dict[str, Any]], *, binary_mode: bool
) -> dict[str, Any] | None:
    if not pruning_results:
        return None
    best: dict[str, Any] | None = None
    best_val = -1.0
    for row in pruning_results:
        met = row.get("metrics") or {}
        _, val = primary_score_from_metrics(met, binary_mode=binary_mode)
        if val > best_val:
            best_val = val
            best = row
    if best is not None:
        key, val = primary_score_from_metrics(best["metrics"], binary_mode=binary_mode)
        best = {**best, "selection_metric": key, "selection_value": float(val)}
    return best


def write_stream_run_summary_md(
    path: Path,
    *,
    binary_mode: bool,
    metrics: dict[str, Any],
    pruning_results: list[dict[str, Any]],
    best_pruning: dict[str, Any] | None,
    dataset_info: dict[str, Any],
    seed: int,
    output_json: Path,
    checkpoint_dense: Path | None,
    checkpoint_pruned: Path | None,
    artifacts_dir: Path | None,
    train_csv: str,
    test_csv: str,
) -> None:
    """One-page team report for train_ereno_stream.py (metrics + pruning + file map)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary_mode:
        write_binary_summary_md(
            path,
            metrics,
            dataset_info=dataset_info,
            seed=seed,
        )
        extra = path.read_text(encoding="utf-8").splitlines()
    else:
        extra = [
            "# Multiclass IDS evaluation (streaming)",
            "",
            f"**Seed:** {seed}",
            "",
            f"| Macro F1 | MCC | Accuracy |",
            f"|---:|---:|---:|",
            f"| {_fmt_float(metrics.get('macro_f1'))} | {_fmt_float(metrics.get('mcc'))} | {_fmt_float(metrics.get('accuracy'))} |",
            "",
        ]

    lines = extra + [
        "",
        "## Pruning sweep (evaluation only unless pruned checkpoint saved)",
        "",
        "Each row reuses the **same trained weights** with global L1 pruning applied in memory. "
        "Metrics are measured on the **full test CSV**.",
        "",
    ]
    if pruning_results:
        score_hdr = "Attack F1" if binary_mode else "Macro F1"
        lines.extend(
            [
                f"| Prune % | {score_hdr} | MCC | FPR | FNR | Sparsity |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pruning_results:
            ratio = int(float(row["target_pruning_ratio"]) * 100)
            met = row.get("metrics") or {}
            sp = row.get("sparsity") or {}
            sp_s = _fmt_float(sp.get("sparsity_ratio"), digits=3) if sp else "n/a"
            if binary_mode:
                lines.append(
                    f"| {ratio} | {_fmt_float(met.get('f1'))} | {_fmt_float(met.get('mcc'))} | "
                    f"{_fmt_float(met.get('false_positive_rate'))} | {_fmt_float(met.get('false_negative_rate'))} | {sp_s} |"
                )
            else:
                lines.append(
                    f"| {ratio} | {_fmt_float(met.get('macro_f1'))} | {_fmt_float(met.get('mcc'))} | — | — | {sp_s} |"
                )
        lines.append("")
        if best_pruning:
            pct = int(float(best_pruning["target_pruning_ratio"]) * 100)
            key = best_pruning.get("selection_metric", "score")
            val = best_pruning.get("selection_value", 0.0)
            lines.extend(
                [
                    f"**Best in sweep:** {pct}% pruning ({key} = {val:.4f} on test set).",
                    "",
                ]
            )
    else:
        lines.append("_No pruning ratios configured (`--pruning-ratios` empty)._")
        lines.append("")

    lines.extend(
        [
            "## Output files (share these paths)",
            "",
            f"| File | Contents |",
            f"|---|---|",
            f"| `{output_json}` | Machine-readable: metrics, GA features, `pruning_sweep` |",
            f"| `{path}` | **This summary** (for the team) |",
        ]
    )
    if checkpoint_dense is not None:
        lines.append(
            f"| `{checkpoint_dense}` | **Trained model (dense / unpruned)** — use for deployment unless you chose a pruned checkpoint |"
        )
    if checkpoint_pruned is not None:
        lines.append(f"| `{checkpoint_pruned}` | **Best pruned model** from the sweep (`--save-best-pruned`) |")
    elif pruning_results and checkpoint_dense is not None:
        lines.append(
            "| _(none)_ | Pruned weights are **not** saved by default; only metrics in JSON. Re-run with `--save-best-pruned` to export the best prune ratio. |"
        )
    if artifacts_dir is not None:
        lines.append(f"| `{artifacts_dir}/` | Plots: ROC, PR, confusion matrix, t-SNE (binary mode) |")
    lines.extend(
        [
            "",
            "## Data used",
            "",
            f"- Train: `{train_csv}`",
            f"- Test: `{test_csv}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_confusion_matrix_ascii(cm: np.ndarray, labels: list[str]) -> str:
    """Fixed-width confusion matrix for terminal (rows=true, cols=pred)."""
    cm = np.asarray(cm, dtype=np.int64)
    col_w = max(8, max((len(s) for s in labels), default=6) + 2)
    header = " " * col_w + "".join(f"{lab:>{col_w}}" for lab in labels)
    lines = [header]
    for i, lab in enumerate(labels):
        row = f"{lab:>{col_w}}" + "".join(f"{int(cm[i, j]):>{col_w}}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def print_epoch_eval_report(
    *,
    epoch: int,
    total_epochs: int,
    eval_scope: str,
    labels: list[str],
    cm: np.ndarray,
    metrics: dict[str, Any],
    binary_mode: bool,
    mean_batch_loss: float,
) -> None:
    """Per-epoch test metrics + confusion matrix for streaming trainer logs."""
    ep = int(epoch)
    tot = int(total_epochs)
    print(f"[stream] epoch {ep:02d}/{tot} mean_batch_loss={mean_batch_loss:.4f}", flush=True)
    print(f"[stream] epoch {ep:02d}/{tot} eval ({eval_scope}):", flush=True)
    if binary_mode:
        print(f"         {format_binary_console_summary(metrics)}", flush=True)
    else:
        print(
            f"         macro_f1={_fmt_float(metrics.get('macro_f1'))} "
            f"mcc={_fmt_float(metrics.get('mcc'))} accuracy={_fmt_float(metrics.get('accuracy'))}",
            flush=True,
        )
    print(f"[stream] epoch {ep:02d}/{tot} confusion_matrix (rows=true, cols=pred):", flush=True)
    for line in format_confusion_matrix_ascii(cm, labels).splitlines():
        print(f"         {line}", flush=True)


def print_stream_run_footer(
    *,
    binary_mode: bool,
    metrics: dict[str, Any],
    best_pruning: dict[str, Any] | None,
    summary_md: Path,
    output_json: Path,
    checkpoint_dense: Path | None,
    checkpoint_pruned: Path | None,
    artifacts_dir: Path | None,
) -> None:
    """Clear end-of-run banner for the terminal."""
    width = 72
    print("\n" + "=" * width, flush=True)
    print("STREAM TRAINING FINISHED", flush=True)
    print("=" * width, flush=True)
    if binary_mode:
        print(f"  Test set (dense model): {format_binary_console_summary(metrics)}", flush=True)
    else:
        print(
            f"  Test set (dense model): macro_f1={_fmt_float(metrics.get('macro_f1'))} "
            f"mcc={_fmt_float(metrics.get('mcc'))}",
            flush=True,
        )
    if best_pruning:
        pct = int(float(best_pruning["target_pruning_ratio"]) * 100)
        key = best_pruning.get("selection_metric", "score")
        val = float(best_pruning.get("selection_value", 0.0))
        bm = best_pruning.get("metrics") or {}
        print(f"  Best pruning: {pct}% ({key}={val:.4f}, mcc={_fmt_float(bm.get('mcc'))})", flush=True)
        if checkpoint_pruned is None:
            print(
                "  Note: pruned weights are metrics-only unless you pass --save-best-pruned",
                flush=True,
            )
    print("", flush=True)
    print("  Read this first:", flush=True)
    print(f"    {summary_md}", flush=True)
    print("  Full numbers:", flush=True)
    print(f"    {output_json}", flush=True)
    if checkpoint_dense is not None:
        print("  Model checkpoint (dense):", flush=True)
        print(f"    {checkpoint_dense}", flush=True)
    if checkpoint_pruned is not None:
        print("  Model checkpoint (best pruned):", flush=True)
        print(f"    {checkpoint_pruned}", flush=True)
    if artifacts_dir is not None:
        print("  Plots:", flush=True)
        print(f"    {artifacts_dir}", flush=True)
    print("=" * width + "\n", flush=True)


def collect_multiclass_labels_stream(
    test_path: Path,
    *,
    resolved_target: str,
    chunk_rows: int,
) -> np.ndarray:
    """Row-aligned multiclass strings in CSV read order (for alignment with scored eval)."""
    import pandas as pd

    labels: list[str] = []
    reader = pd.read_csv(test_path, chunksize=chunk_rows)
    for chunk in reader:
        raw = chunk[resolved_target].astype(str).str.strip()
        labels.extend(raw.tolist())
    return np.asarray(labels, dtype=object)


def save_selected_features(path: Path, ga_summary: dict[str, Any], feature_names: list[str]) -> None:
    selected = list(ga_summary.get("selected_features") or [])
    if not selected and feature_names:
        selected = list(feature_names)
    payload = {
        "selected_features": selected,
        "selected_feature_count": len(selected),
        "protected_feature_names": list(ga_summary.get("protected_feature_names") or []),
        "ga_best_fitness": ga_summary.get("best_fitness"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred normal", "pred attack"])
    ax.set_yticks([0, 1], labels=["true normal", "true attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=12)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, *, title: str) -> None:
    y_bin = (np.asarray(y_true) == ATTACK_CLASS_ID).astype(np.int64)
    fpr, tpr, _ = roc_curve(y_bin, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.plot(fpr, tpr, lw=2, label=f"attack (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, *, title: str) -> None:
    y_bin = (np.asarray(y_true) == ATTACK_CLASS_ID).astype(np.int64)
    prec, rec, _ = precision_recall_curve(y_bin, y_score)
    ap = average_precision_score(y_bin, y_score)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.plot(rec, prec, lw=2, label=f"attack (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tsne_binary(
    embeddings: np.ndarray,
    y_binary: np.ndarray,
    out_path: Path,
    *,
    title: str,
    seed: int = 42,
    perplexity: float = 30.0,
    max_rows: int = 8000,
) -> None:
    from sklearn.manifold import TSNE

    n = len(embeddings)
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        embeddings = embeddings[idx]
        y_binary = y_binary[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (len(embeddings) - 1) / 3)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    proj = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for cid, name, color in (
        (NORMAL_CLASS_ID, "normal", "#2ca02c"),
        (ATTACK_CLASS_ID, "attack", "#d62728"),
    ):
        mask = y_binary == cid
        if not mask.any():
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5, c=color, label=f"{name} (n={int(mask.sum())})", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
