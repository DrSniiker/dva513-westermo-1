"""
Feature-level diagnostics for the streaming ERENO pipeline.

Produces, for an encoded class-stratified reservoir:

1. Per-feature importance per class (one-vs-rest ANOVA F-stat, plus a
   centroid-z separability score).
2. Class centroids and pairwise centroid distance matrix (Euclidean on
   standardized features).
3. Pairwise feature-overlap analysis between user-chosen classes (per feature,
   |mu_a - mu_b| / sqrt(var_a + var_b)).
4. A confusion-driven feature ranking that highlights features useful for
   `inverse_replay` and `masquerade_fake_fault` against their main confusers
   (`normal`, `poisoned_high_rate`, `masquerade_fake_normal`).

Outputs JSON; prints a short summary to stdout.

Example:
  python src/modeling/feature_diagnostics.py \\
    --train-csv outputs/train_real.csv \\
    --reservoir-rows 60000 \\
    --top-k 12 \\
    --pairs "inverse_replay:normal,inverse_replay:poisoned_high_rate,masquerade_fake_fault:normal,masquerade_fake_fault:poisoned_high_rate" \\
    --output-json reports/feature_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modeling.train_ereno_stream import (  # noqa: E402
    _feature_names,
    _pass1_train_stream,
    _reservoir_stratified_stream,
    _stratified_quotas,
)


def _zstandardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def _per_class_importance(
    x: np.ndarray, y: np.ndarray, labels: list[str], top_k: int
) -> dict[str, dict[str, list]]:
    """One-vs-rest ANOVA F per class (binary problem 'class c vs the rest')."""
    results: dict[str, dict[str, list]] = {}
    classes_in_pool = np.unique(y)
    for cid in classes_in_pool:
        cls = labels[int(cid)]
        y_bin = (y == cid).astype(np.int64)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        with np.errstate(invalid="ignore"):
            f_stat, _ = f_classif(x, y_bin)
        f_stat = np.nan_to_num(f_stat, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-f_stat)
        results[cls] = {
            "feature_idx": [int(i) for i in order[:top_k].tolist()],
            "f_stat": [float(f_stat[i]) for i in order[:top_k].tolist()],
        }
    return results


def _class_centroids(
    x: np.ndarray, y: np.ndarray, labels: list[str]
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
    """Return mean vector per class and the pairwise Euclidean distance dict."""
    centroids: dict[str, np.ndarray] = {}
    for cid in np.unique(y):
        rows = x[y == cid]
        if len(rows) == 0:
            continue
        centroids[labels[int(cid)]] = rows.mean(axis=0)
    centroid_table = {k: v.tolist() for k, v in centroids.items()}
    keys = sorted(centroids.keys())
    pair_dist: dict[str, dict[str, float]] = {a: {} for a in keys}
    for a in keys:
        for b in keys:
            if a == b:
                pair_dist[a][b] = 0.0
            else:
                d = float(np.linalg.norm(centroids[a] - centroids[b]))
                pair_dist[a][b] = d
    return centroid_table, pair_dist


def _separability_per_feature(
    x: np.ndarray, y: np.ndarray, cid_a: int, cid_b: int
) -> np.ndarray:
    """Per-feature |mu_a - mu_b| / sqrt(var_a + var_b). Higher = more separable."""
    a = x[y == cid_a]
    b = x[y == cid_b]
    if len(a) == 0 or len(b) == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    va, vb = a.var(axis=0), b.var(axis=0)
    denom = np.sqrt(np.maximum(va + vb, 1e-12))
    return np.abs(mu_a - mu_b) / denom


def _pairwise_overlap(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label_to_id: dict[str, int],
    pairs: list[tuple[str, str]],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    """For each (a, b), top-k most-discriminating features and a global overlap score."""
    out: dict[str, dict[str, Any]] = {}
    for a, b in pairs:
        if a not in label_to_id or b not in label_to_id:
            continue
        sep = _separability_per_feature(x, y, label_to_id[a], label_to_id[b])
        order = np.argsort(-sep)
        top_idx = [int(i) for i in order[:top_k]]
        bottom_idx = [int(i) for i in order[-top_k:]]
        out[f"{a}__vs__{b}"] = {
            "top_discriminating": [
                {"feature": feature_names[i], "score": float(sep[i])}
                for i in top_idx
            ],
            "least_discriminating": [
                {"feature": feature_names[i], "score": float(sep[i])}
                for i in bottom_idx
            ],
            "mean_separability": float(np.mean(sep)),
            "median_separability": float(np.median(sep)),
        }
    return out


def _confusion_driven_ranking(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label_to_id: dict[str, int],
    target_classes: list[str],
    confuser_classes: list[str],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """For each target class, pick features that discriminate it from each listed confuser."""
    out: dict[str, list[dict[str, Any]]] = {}
    for tcls in target_classes:
        if tcls not in label_to_id:
            continue
        scores = np.zeros(x.shape[1], dtype=np.float64)
        per_pair_scores: dict[str, np.ndarray] = {}
        for ccls in confuser_classes:
            if ccls == tcls or ccls not in label_to_id:
                continue
            sep = _separability_per_feature(x, y, label_to_id[tcls], label_to_id[ccls])
            per_pair_scores[ccls] = sep
            scores += sep
        if not per_pair_scores:
            continue
        order = np.argsort(-scores)[:top_k]
        out[tcls] = [
            {
                "feature": feature_names[i],
                "total_score": float(scores[i]),
                "per_confuser": {c: float(per_pair_scores[c][i]) for c in per_pair_scores},
            }
            for i in order
        ]
    return out


def _parse_pairs(spec: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"--pairs token '{tok}' must look like 'classA:classB'.")
        a, b = tok.split(":", 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            raise ValueError(f"--pairs token '{tok}' has empty class name.")
        pairs.append((a, b))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature diagnostics on a class-stratified reservoir.")
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--target-col", type=str, default="class")
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--reservoir-rows", type=int, default=60_000)
    p.add_argument("--variance-threshold", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument(
        "--pairs",
        type=str,
        default="inverse_replay:normal,inverse_replay:poisoned_high_rate,"
        "inverse_replay:masquerade_fake_fault,"
        "masquerade_fake_fault:normal,masquerade_fake_fault:poisoned_high_rate,"
        "masquerade_fake_normal:normal",
        help="Comma-separated 'a:b' pairs for overlap analysis.",
    )
    p.add_argument(
        "--target-classes",
        type=str,
        default="inverse_replay,masquerade_fake_fault",
        help="Confusion-driven ranking targets.",
    )
    p.add_argument(
        "--confuser-classes",
        type=str,
        default="normal,poisoned_high_rate,masquerade_fake_normal,inverse_replay,masquerade_fake_fault",
        help="Pool of potential confusers; the target itself is auto-skipped.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path, default=Path("reports/feature_diagnostics.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    print("[diag] pass1: stats + class counts...", flush=True)
    state, n_train, class_counts = _pass1_train_stream(
        args.train_csv, args.target_col, args.chunk_rows, args.variance_threshold
    )
    print(f"[diag] train_rows={n_train} labels={state.num_classes}", flush=True)
    quotas = _stratified_quotas(class_counts, state.labels, args.reservoir_rows)
    quota_summary = {state.labels[c]: int(n) for c, n in sorted(quotas.items())}
    print(f"[diag] reservoir quotas: {quota_summary}", flush=True)
    x, y = _reservoir_stratified_stream(args.train_csv, state, args.chunk_rows, quotas, rng)
    print(f"[diag] reservoir rows={len(x)} dim={x.shape[1]}", flush=True)
    feature_names = _feature_names(state)
    xz = _zstandardize(x)

    importance = _per_class_importance(xz, y, state.labels, args.top_k)
    centroids, pair_dist = _class_centroids(xz, y, state.labels)
    pair_specs = _parse_pairs(args.pairs)
    overlap = _pairwise_overlap(
        xz, y, feature_names, state.label_to_id, pair_specs, args.top_k
    )
    target_classes = [s.strip() for s in args.target_classes.split(",") if s.strip()]
    confuser_classes = [s.strip() for s in args.confuser_classes.split(",") if s.strip()]
    confusion_rank = _confusion_driven_ranking(
        xz, y, feature_names, state.label_to_id, target_classes, confuser_classes, args.top_k
    )

    payload: dict[str, Any] = {
        "train_csv": str(args.train_csv),
        "reservoir_rows": int(len(x)),
        "feature_count": int(x.shape[1]),
        "labels": state.labels,
        "reservoir_quotas": quota_summary,
        "per_class_one_vs_rest_importance": {
            cls: {
                "features": [feature_names[i] for i in info["feature_idx"]],
                "f_stat": info["f_stat"],
            }
            for cls, info in importance.items()
        },
        "class_centroids": centroids,
        "pairwise_centroid_distance": pair_dist,
        "pairwise_feature_overlap": overlap,
        "confusion_driven_ranking": confusion_rank,
        "feature_names": feature_names,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[diag] wrote {args.output_json}", flush=True)

    if confusion_rank:
        print("[diag] confusion-driven top features:", flush=True)
        for tcls, items in confusion_rank.items():
            head = ", ".join(f"{it['feature']}({it['total_score']:.2f})" for it in items[:8])
            print(f"  {tcls}: {head}", flush=True)
    if pair_dist:
        keys = sorted(pair_dist.keys())
        width = max(8, max(len(k) for k in keys))
        print("[diag] centroid distance matrix (rows/cols=class):", flush=True)
        header = " " * (width + 2) + "  ".join(f"{k:>{width}}" for k in keys)
        print(header, flush=True)
        for a in keys:
            row = "  ".join(f"{pair_dist[a][b]:>{width}.2f}" for b in keys)
            print(f"{a:>{width}}  {row}", flush=True)


if __name__ == "__main__":
    main()
