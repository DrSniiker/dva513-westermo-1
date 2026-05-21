# `flow_dpi` (C) — not wired into the Python ERENO trainer

This directory contains a **standalone C library** (`flow_meter.c`) for **IPv4 / TCP / UDP** flow hashing and metering (5-tuple style: IP addresses, ports, protocol).

## Why it is separate from `train_ereno.py` / `train_ereno_ann.py`

- The ERENO IEC-61850 pipeline in Python consumes **tabular CSV** features (GOOSE-oriented columns such as `gooseAppid`, `SqNum`, `StNum`, …) produced **upstream** of this repo.
- `flow_meter.c` operates on **raw Ethernet/IP frames**, not on the ERENO preprocessed schema. Feeding it would require a **PCAP ingest + feature extraction** path that reproduces the same semantics as your CSVs — that is a different subsystem (and typically **edge / embedded** work), not a drop-in replacement for `iec60/flow_aggregate.py`.

## What to use for “flows” in the ANN pipeline

Use the **Python** path, which is now first-class:

1. Row-level CSVs (`train.csv` / `test.csv`).
2. `python src/iec60/build_ereno_flow_splits.py` → `outputs/train_flows.csv`, `outputs/test_flows.csv`.
3. `python src/modeling/train_ereno.py --representation flow --use-official-test`.

## When `flow_dpi` *does* make sense

- Custom firmware or a Westermo-class device that meters **IP flows on the wire** before any ML.
- Benchmarking C throughput vs Python batch jobs (separate experiment).

If you later add a small **PCAP → CSV** exporter that emits columns compatible with `flow_aggregate.py`, you could document it here and optionally call into `flow_meter` from C; that remains **future work**.
