"""
ERENO IEC-61850: aggregate preprocessed *row* (packet/event) CSVs into *flow*-level tables.

Used by `src/modeling/ereno_to_flows.py` (CLI) and `src/iec60/build_ereno_flow_splits.py` (train+test batch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FlowConfig:
    inactive_timeout_s: float = 1.0
    active_timeout_s: float = 5.0
    sampling_rate: int = 1


KNOWN_ATTACKS = {
    "random_replay",
    "inverse_replay",
    "masquerade_fake_fault",
    "masquerade_fake_normal",
    "injection",
    "high_stnum",
    "poisoned_high_rate",
}


def recover_class(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "class" in out.columns and out["class"].notna().any():
        out["class"] = out["class"].astype(str).str.strip()
        return out
    if "delay" in out.columns:
        out["class"] = out["delay"].astype(str).str.strip()
    else:
        out["class"] = "normal"
    return out


def choose_label(labels: pd.Series) -> str:
    lbl = labels.astype(str).str.strip().str.lower()
    attack_rows = lbl[lbl.isin(KNOWN_ATTACKS)]
    if not attack_rows.empty:
        return attack_rows.iloc[0]
    if not lbl.empty:
        return lbl.mode(dropna=True).iloc[0]
    return "normal"


def build_flows(df: pd.DataFrame, cfg: FlowConfig, seed: int) -> pd.DataFrame:
    work = recover_class(df)
    for col in ("ethSrc", "ethDst", "gooseAppid", "protocol", "SqNum", "StNum", "frameLen", "Time", "timestampDiff"):
        if col not in work.columns:
            work[col] = 0

    if cfg.sampling_rate > 1:
        rng = np.random.default_rng(seed)
        keep = rng.integers(0, cfg.sampling_rate, len(work)) == 0
        work = work.loc[keep].copy()

    work["ts"] = pd.to_numeric(work["Time"], errors="coerce").ffill().fillna(0.0)
    work["frameLen"] = pd.to_numeric(work["frameLen"], errors="coerce").fillna(0.0)
    work["SqNum"] = pd.to_numeric(work["SqNum"], errors="coerce").fillna(-1)
    work["StNum"] = pd.to_numeric(work["StNum"], errors="coerce").fillna(-1)
    work["timestampDiff"] = pd.to_numeric(work["timestampDiff"], errors="coerce").fillna(0.0)

    key_cols = ["ethSrc", "ethDst", "gooseAppid", "protocol"]
    work = work.sort_values(key_cols + ["ts"]).reset_index(drop=True)

    key_changed = (work[key_cols] != work[key_cols].shift(1)).any(axis=1)
    dt = work["ts"] - work["ts"].shift(1).fillna(work["ts"])
    inactive_break = dt > cfg.inactive_timeout_s
    active_age = work["ts"] - work.groupby(key_cols)["ts"].transform("first")
    active_break = active_age > cfg.active_timeout_s
    new_flow = key_changed | inactive_break | active_break
    work["flow_id"] = new_flow.cumsum().astype(np.int64)

    prev_sq = work.groupby(key_cols + ["flow_id"])["SqNum"].shift(1)
    prev_st = work.groupby(key_cols + ["flow_id"])["StNum"].shift(1)
    work["sq_out_of_order"] = ((prev_sq.notna()) & (work["SqNum"] < prev_sq)).astype(np.int32)
    work["st_transitions"] = ((prev_st.notna()) & (work["StNum"] != prev_st)).astype(np.int32)
    work["bit_string_hits"] = work.astype(str).apply(
        lambda r: int("bit_string" in " ".join(r.values).lower()),
        axis=1,
    )
    work["ber_padding_anomaly_hits"] = 0

    grouped = work.groupby(key_cols + ["flow_id"], sort=False)
    out = grouped.agg(
        flow_start=("ts", "min"),
        flow_end=("ts", "max"),
        packets_total=("ts", "count"),
        bytes_total=("frameLen", "sum"),
        min_pkt_len=("frameLen", "min"),
        max_pkt_len=("frameLen", "max"),
        avg_pkt_len=("frameLen", "mean"),
        avg_timestamp_diff=("timestampDiff", "mean"),
        stnum_transitions=("st_transitions", "sum"),
        sqnum_out_of_order=("sq_out_of_order", "sum"),
        bit_string_hits=("bit_string_hits", "sum"),
        ber_padding_anomaly_hits=("ber_padding_anomaly_hits", "sum"),
        label=("class", choose_label),
    ).reset_index()

    out["duration"] = (out["flow_end"] - out["flow_start"]).clip(lower=1e-6)
    out["pkt_rate_per_sec"] = out["packets_total"] / out["duration"]
    out["byte_rate_per_sec"] = out["bytes_total"] / out["duration"]
    out["class"] = out["label"]
    out = out.drop(columns=["label"])
    return out
