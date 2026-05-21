"""
ERENO tabular preprocessing: duplicate rows and low-variance numeric feature removal.

Variance is computed on the **fit** split only; the same columns are dropped from **eval**
(no test-set statistics are used for selection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_duplicate_removal(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows (all columns). Within one split only — no cross-split leakage."""
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, int(n_before - len(out))


def apply_variance_feature_filter(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Drop numeric columns whose sample variance on **fit_df** is <= threshold.
    Non-numeric columns are always kept. If that would remove every numeric column,
    keep the single numeric column with the largest variance on fit_df.
    """
    if threshold < 0:
        raise ValueError("variance threshold must be >= 0")

    feature_cols = [c for c in fit_df.columns if c != target_col]
    if not feature_cols:
        return fit_df, eval_df, []

    numeric_cols = fit_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feature_cols if c not in numeric_cols]

    dropped: list[str] = []
    keep_numeric: list[str] = []
    variances: dict[str, float] = {}
    for c in numeric_cols:
        v = float(fit_df[c].var(skipna=True))
        if np.isnan(v):
            v = 0.0
        variances[c] = v
        if v > threshold:
            keep_numeric.append(c)
        else:
            dropped.append(c)

    if not keep_numeric and numeric_cols:
        best = max(numeric_cols, key=lambda col: variances.get(col, 0.0))
        dropped = [c for c in numeric_cols if c != best]
        keep_numeric = [best]

    # Drop columns instead of `[cols].copy()`: on multi-million-row frames a deep copy
    # of a column subset can still allocate ~hundreds of MiB and fail on typical laptops.
    cols_to_drop = [c for c in dropped if c in fit_df.columns]
    missing_drop = [c for c in cols_to_drop if c not in eval_df.columns]
    if missing_drop:
        raise ValueError(f"eval split missing columns expected for variance drop: {missing_drop}")
    fit_out = fit_df.drop(columns=cols_to_drop, errors="raise")
    eval_out = eval_df.drop(columns=cols_to_drop, errors="raise")
    return fit_out, eval_out, dropped
