"""Detect WEKA ARFF files mislabeled as .csv (common with ERENO releases)."""

from __future__ import annotations

from pathlib import Path


def file_looks_like_weka_arff(path: Path, read_bytes: int = 4096) -> bool:
    with path.open("rb") as f:
        head = f.read(read_bytes).lstrip()
    if not head:
        return False
    low = head.lower()
    return low.startswith(b"@relation") or b"\n@relation" in low[: read_bytes // 2]


def assert_comma_separated_csv(path: Path, *, role: str) -> None:
    if not file_looks_like_weka_arff(path):
        return
    raise ValueError(
        f"{role} path {path} looks like WEKA ARFF (starts with @relation / @attribute block), "
        "not comma-separated CSV. The training scripts require real CSV.\n"
        "Fix: export CSV from WEKA (Save as CSV), or rename to .arff and convert with a tool that "
        "writes comma-separated UTF-8 CSV, then run: python src/iec60/publish_splits.py --train-source ... --test-source ..."
    )
