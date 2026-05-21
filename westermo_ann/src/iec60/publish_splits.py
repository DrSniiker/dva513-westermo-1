"""
Copy ERENO IEC-61850 train/test CSVs into the repository root as train.csv / test.csv.

This is the bridge between *dataset artifacts* (preprocessed tables) and the modeling
entry points (`train_ereno.py`, `train_ereno_stream.py`, `train_ereno_ann.py`), which expect those names
by default under REPO_ROOT.

 PCAP parsing is not performed here; see src/iec60/README.md for the full chain.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from paths import DEFAULT_PUBLISH_TEST, DEFAULT_PUBLISH_TRAIN, REPO_ROOT
from tabular_guard import assert_comma_separated_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish ERENO train/test CSVs to repo root (train.csv / test.csv).")
    p.add_argument(
        "--train-source",
        type=Path,
        default=DEFAULT_PUBLISH_TRAIN,
        help=f"Input train CSV (default: {DEFAULT_PUBLISH_TRAIN})",
    )
    p.add_argument(
        "--test-source",
        type=Path,
        default=DEFAULT_PUBLISH_TEST,
        help=f"Input test CSV (default: {DEFAULT_PUBLISH_TEST})",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        default=REPO_ROOT,
        help=f"Directory to write train.csv and test.csv (default: {REPO_ROOT})",
    )
    p.add_argument("--dry-run", action="store_true", help="Only validate sources and print destinations.")
    return p.parse_args()


def _peek_header(path: Path) -> list[str]:
    assert_comma_separated_csv(path, role="Source")
    return list(pd.read_csv(path, nrows=0).columns)


def main() -> None:
    args = parse_args()
    train_src = args.train_source.resolve()
    test_src = args.test_source.resolve()
    dest = args.dest_dir.resolve()

    if not train_src.is_file():
        raise FileNotFoundError(f"Train source not found: {train_src}")
    if not test_src.is_file():
        raise FileNotFoundError(f"Test source not found: {test_src}")

    train_cols = set(_peek_header(train_src))
    test_cols = set(_peek_header(test_src))
    if train_cols != test_cols:
        print("Warning: train and test column names differ.")
        print(f"  only in train: {sorted(train_cols - test_cols)[:20]}{'...' if len(train_cols - test_cols) > 20 else ''}")
        print(f"  only in test:  {sorted(test_cols - train_cols)[:20]}{'...' if len(test_cols - train_cols) > 20 else ''}")

    out_train = dest / "train.csv"
    out_test = dest / "test.csv"
    print(f"Source train: {train_src}")
    print(f"Source test:  {test_src}")
    print(f"Dest train:   {out_train}")
    print(f"Dest test:    {out_test}")

    if args.dry_run:
        print("Dry run: no files copied.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_src, out_train)
    shutil.copy2(test_src, out_test)
    print("Published train.csv and test.csv.")


if __name__ == "__main__":
    main()
