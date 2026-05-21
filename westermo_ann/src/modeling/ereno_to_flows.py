"""
CLI: aggregate one ERENO row-level CSV into flow-level CSV.

Implementation lives in `src/iec60/flow_aggregate.py` (dataset-specific). For train+test
batch generation use `python src/iec60/build_ereno_flow_splits.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from iec60.flow_aggregate import FlowConfig, build_flows
from iec60.tabular_guard import assert_comma_separated_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate ERENO rows into flow-level records")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--inactive-timeout", type=float, default=1.0)
    p.add_argument("--active-timeout", type=float, default=5.0)
    p.add_argument("--sampling-rate", type=int, default=1, help="1=keep all, 10=keep ~10%")
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_comma_separated_csv(args.input_csv, role="Input")
    cfg = FlowConfig(
        inactive_timeout_s=args.inactive_timeout,
        active_timeout_s=args.active_timeout,
        sampling_rate=max(1, args.sampling_rate),
    )
    df = pd.read_csv(args.input_csv)
    flows = build_flows(df, cfg=cfg, seed=args.random_seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    flows.to_csv(args.output_csv, index=False)
    print(f"Input rows: {len(df)}")
    print(f"Output flows: {len(flows)}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
