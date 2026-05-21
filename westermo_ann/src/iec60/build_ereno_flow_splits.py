"""
Build train/test *flow* CSVs from ERENO *row*-level train/test CSVs.

Writes default outputs:
  outputs/train_flows.csv
  outputs/test_flows.csv

Then train with:
  python src/modeling/train_ereno.py --representation flow --use-official-test
(or pass explicit --train-csv / --test-csv paths to the flow files.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from flow_aggregate import FlowConfig, build_flows
from paths import FLOW_TEST_CSV, FLOW_TRAIN_CSV, REPO_ROOT, REPO_TEST_CSV, REPO_TRAIN_CSV
from tabular_guard import assert_comma_separated_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ERENO flow-level train/test CSVs from row-level CSVs.")
    p.add_argument("--train-rows", type=Path, default=REPO_TRAIN_CSV, help="Row-level train CSV")
    p.add_argument("--test-rows", type=Path, default=REPO_TEST_CSV, help="Row-level test CSV")
    p.add_argument("--output-train-flows", type=Path, default=FLOW_TRAIN_CSV, help="Flow-level train output")
    p.add_argument("--output-test-flows", type=Path, default=FLOW_TEST_CSV, help="Flow-level test output")
    p.add_argument("--inactive-timeout", type=float, default=1.0)
    p.add_argument("--active-timeout", type=float, default=5.0)
    p.add_argument("--sampling-rate", type=int, default=1, help="1=all rows, N=keep ~1/N")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _one_split(path: Path, cfg: FlowConfig, seed: int, out_path: Path, label: str) -> None:
    assert_comma_separated_csv(path, role=label)
    df = pd.read_csv(path)
    flows = build_flows(df, cfg=cfg, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flows.to_csv(out_path, index=False)
    print(f"[{label}] rows_in={len(df)} flows_out={len(flows)} -> {out_path}")


def main() -> None:
    args = parse_args()
    cfg = FlowConfig(
        inactive_timeout_s=args.inactive_timeout,
        active_timeout_s=args.active_timeout,
        sampling_rate=max(1, args.sampling_rate),
    )
    _one_split(args.train_rows, cfg, args.seed, args.output_train_flows, "train")
    _one_split(args.test_rows, cfg, args.seed + 1, args.output_test_flows, "test")
    print("Done. Use --representation flow with train_ereno.py, or --train-csv/--test-csv pointing here.")


if __name__ == "__main__":
    main()
