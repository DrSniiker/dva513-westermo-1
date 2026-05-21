"""ERENO IDS label helpers: optional collapse of multiclass names to binary (normal=0, attack=1)."""

from __future__ import annotations

from collections import Counter

BINARY_LABELS: tuple[str, str] = ("normal", "attack")
NORMAL_CLASS_NAME = "normal"
ATTACK_CLASS_NAME = "attack"
NORMAL_CLASS_ID = 0
ATTACK_CLASS_ID = 1


def normalize_label_name(label: str) -> str:
    return str(label).strip()


def is_normal_label(label: str) -> bool:
    return normalize_label_name(label) == NORMAL_CLASS_NAME


def multiclass_label_to_binary_id(label: str) -> int:
    """Map a multiclass name to 0 (normal) or 1 (any other class)."""
    return NORMAL_CLASS_ID if is_normal_label(label) else ATTACK_CLASS_ID


def binary_counts_from_multiclass(class_counts: Counter | dict[str, int]) -> dict[str, int]:
    """Aggregate per-class counts into normal(0) vs attack(1)."""
    normal_n = 0
    attack_n = 0
    for lab, n in class_counts.items():
        cnt = int(n)
        if is_normal_label(str(lab)):
            normal_n += cnt
        else:
            attack_n += cnt
    return {NORMAL_CLASS_NAME: normal_n, ATTACK_CLASS_NAME: attack_n}


def task_type_for_mode(binary_mode: bool) -> str:
    return "binary" if binary_mode else "multiclass"


def format_binary_count_line(counts: dict[str, int]) -> str:
    return (
        f"normal(0)={counts.get(NORMAL_CLASS_NAME, 0)} "
        f"attack(1)={counts.get(ATTACK_CLASS_NAME, 0)}"
    )


def stream_binary_state_fields() -> tuple[list[str], dict[str, int], int]:
    """Fixed label order and ids for streaming encoder binary mode."""
    labels = list(BINARY_LABELS)
    label_to_id = {NORMAL_CLASS_NAME: NORMAL_CLASS_ID, ATTACK_CLASS_NAME: ATTACK_CLASS_ID}
    return labels, label_to_id, 2


def count_binary_labels_in_csv(
    csv_path,
    *,
    target_col: str,
    chunk_rows: int,
    resolve_label_column,
) -> tuple[Counter, int]:
    """
    One pass over a CSV: multiclass histogram + row count (for train/test distribution checks).
    `resolve_label_column(first_df, target_col) -> str` matches the streaming trainer.
    """
    import pandas as pd
    from pathlib import Path

    path = Path(csv_path)
    reader = pd.read_csv(path, chunksize=chunk_rows)
    first = next(reader)
    if target_col not in first.columns:
        raise ValueError(f"Missing {target_col!r} in {path}.")
    resolved = resolve_label_column(first, target_col)
    if resolved not in first.columns:
        raise ValueError(f"Resolved label column {resolved!r} missing from {path}.")

    class_counts: Counter = Counter()
    n_rows = 0

    def consume(batch) -> None:
        nonlocal n_rows
        n_rows += len(batch)
        class_counts.update(batch[resolved].astype(str).str.strip().tolist())

    consume(first)
    for chunk in reader:
        consume(chunk)
    return class_counts, n_rows
