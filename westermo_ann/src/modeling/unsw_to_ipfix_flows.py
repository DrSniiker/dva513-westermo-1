from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FlowConfig:
    inactive_timeout_s: float = 15.0
    active_timeout_s: float = 60.0
    sampling_rate: int = 1
    min_packet_bytes: int = 0


FLOW_KEY_COLUMNS = ["srcip", "dstip", "dsport", "proto"]
OUTPUT_COLUMNS = [
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "flow_start_ms",
    "flow_end_ms",
    "duration_ms",
    "packets_total",
    "bytes_total",
    "byte_rate_per_sec",
    "pkt_rate_per_sec",
    "avg_pkt_len",
    "min_pkt_len",
    "max_pkt_len",
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_fin_count",
    "tcp_rst_count",
    "tcp_psh_count",
    "tcp_urg_count",
    "stnum_transitions",
    "sqnum_out_of_order",
    "bit_string_hits",
    "ber_padding_anomaly_hits",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UNSW-NB15 packet rows to IPFIX-like flow records")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--inactive-timeout", type=float, default=15.0)
    parser.add_argument("--active-timeout", type=float, default=60.0)
    parser.add_argument("--sampling-rate", type=int, default=1, help="1=keep all, 10=keep about 10%")
    parser.add_argument("--min-packet-bytes", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _safe_time_to_ms(df: pd.DataFrame) -> pd.Series:
    if "stime" in df.columns:
        ts = pd.to_numeric(df["stime"], errors="coerce")
    elif "frame.time_epoch" in df.columns:
        ts = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    else:
        ts = pd.Series(np.arange(len(df), dtype=np.float64), index=df.index)
    return (ts.fillna(method="ffill").fillna(0.0) * 1000.0).astype(np.int64)


def _extract_tcp_flags(df: pd.DataFrame) -> pd.DataFrame:
    state = df.get("state", pd.Series([""] * len(df), index=df.index)).astype(str).str.upper()
    return pd.DataFrame(
        {
            "tcp_syn_count": state.str.contains("SYN", regex=False).astype(np.int32),
            "tcp_ack_count": state.str.contains("ACK", regex=False).astype(np.int32),
            "tcp_fin_count": state.str.contains("FIN", regex=False).astype(np.int32),
            "tcp_rst_count": state.str.contains("RST", regex=False).astype(np.int32),
            "tcp_psh_count": state.str.contains("PSH", regex=False).astype(np.int32),
            "tcp_urg_count": state.str.contains("URG", regex=False).astype(np.int32),
        }
    )


def _sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    stnum = pd.to_numeric(df.get("stNum", df.get("stnum", 0)), errors="coerce").fillna(-1).astype(np.int64)
    sqnum = pd.to_numeric(df.get("sqNum", df.get("sqnum", 0)), errors="coerce").fillna(-1).astype(np.int64)
    prev_stnum = stnum.shift(1).fillna(stnum)
    prev_sqnum = sqnum.shift(1).fillna(sqnum)
    return pd.DataFrame(
        {
            "stnum_transition": (stnum != prev_stnum).astype(np.int32),
            "sqnum_out_of_order": (sqnum < prev_sqnum).astype(np.int32),
        }
    )


def _payload_indicators(df: pd.DataFrame) -> pd.DataFrame:
    payload = df.get("payload", pd.Series([""] * len(df), index=df.index)).astype(str)
    bit_hit = payload.str.contains("bit_string", case=False, regex=False).astype(np.int32)

    # Practical placeholder for BER BIT STRING padding anomaly.
    # If explicit column exists, use it; otherwise derive from payload marker.
    if "ber_bitstring_padding" in df.columns:
        pad = pd.to_numeric(df["ber_bitstring_padding"], errors="coerce").fillna(0).astype(np.int32)
        ber_anomaly = ((pad < 0) | (pad > 7)).astype(np.int32)
    else:
        ber_anomaly = payload.str.contains("ber_bitstring_padding_anomaly", case=False, regex=False).astype(np.int32)

    return pd.DataFrame(
        {"bit_string_hits": bit_hit, "ber_padding_anomaly_hits": ber_anomaly}
    )


def build_flow_records(packet_df: pd.DataFrame, cfg: FlowConfig, random_seed: int) -> pd.DataFrame:
    df = packet_df.copy()
    for required in FLOW_KEY_COLUMNS:
        if required not in df.columns:
            raise ValueError(f"Missing required flow-key column: {required}")

    df["ts_ms"] = _safe_time_to_ms(df)
    df["pkt_len"] = pd.to_numeric(df.get("sbytes", df.get("len", 0)), errors="coerce").fillna(0).astype(np.int64)
    df = df[df["pkt_len"] >= int(cfg.min_packet_bytes)].copy()

    if cfg.sampling_rate > 1:
        rng = np.random.default_rng(seed=random_seed)
        keep_mask = rng.integers(0, cfg.sampling_rate, size=len(df)) == 0
        df = df[keep_mask].copy()

    flag_df = _extract_tcp_flags(df)
    seq_df = _sequence_features(df)
    payload_df = _payload_indicators(df)
    df = pd.concat([df.reset_index(drop=True), flag_df, seq_df, payload_df], axis=1)
    df = df.sort_values(FLOW_KEY_COLUMNS + ["ts_ms"]).reset_index(drop=True)

    key_change = (df[FLOW_KEY_COLUMNS] != df[FLOW_KEY_COLUMNS].shift(1)).any(axis=1)
    delta_ms = df["ts_ms"] - df["ts_ms"].shift(1).fillna(df["ts_ms"])
    inactive_break = delta_ms > int(cfg.inactive_timeout_s * 1000.0)
    active_ref = df["ts_ms"] - df.groupby(FLOW_KEY_COLUMNS)["ts_ms"].transform("first")
    active_break = active_ref > int(cfg.active_timeout_s * 1000.0)
    new_flow = key_change | inactive_break | active_break
    df["flow_local_id"] = new_flow.cumsum().astype(np.int64)

    grouped = df.groupby(FLOW_KEY_COLUMNS + ["flow_local_id"], sort=False)
    out = grouped.agg(
        flow_start_ms=("ts_ms", "min"),
        flow_end_ms=("ts_ms", "max"),
        packets_total=("ts_ms", "count"),
        bytes_total=("pkt_len", "sum"),
        min_pkt_len=("pkt_len", "min"),
        max_pkt_len=("pkt_len", "max"),
        avg_pkt_len=("pkt_len", "mean"),
        tcp_syn_count=("tcp_syn_count", "sum"),
        tcp_ack_count=("tcp_ack_count", "sum"),
        tcp_fin_count=("tcp_fin_count", "sum"),
        tcp_rst_count=("tcp_rst_count", "sum"),
        tcp_psh_count=("tcp_psh_count", "sum"),
        tcp_urg_count=("tcp_urg_count", "sum"),
        stnum_transitions=("stnum_transition", "sum"),
        sqnum_out_of_order=("sqnum_out_of_order", "sum"),
        bit_string_hits=("bit_string_hits", "sum"),
        ber_padding_anomaly_hits=("ber_padding_anomaly_hits", "sum"),
        label=("label", "max"),
    ).reset_index()

    out["duration_ms"] = (out["flow_end_ms"] - out["flow_start_ms"]).clip(lower=1)
    duration_s = out["duration_ms"] / 1000.0
    out["byte_rate_per_sec"] = out["bytes_total"] / duration_s
    out["pkt_rate_per_sec"] = out["packets_total"] / duration_s
    out = out.rename(
        columns={
            "srcip": "src_ip",
            "dstip": "dst_ip",
            "dsport": "dst_port",
            "proto": "protocol",
        }
    )
    return out[OUTPUT_COLUMNS].sort_values(["flow_start_ms", "src_ip", "dst_ip"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = FlowConfig(
        inactive_timeout_s=args.inactive_timeout,
        active_timeout_s=args.active_timeout,
        sampling_rate=max(1, args.sampling_rate),
        min_packet_bytes=max(0, args.min_packet_bytes),
    )

    in_df = pd.read_csv(args.input_csv)
    out_df = build_flow_records(in_df, cfg, random_seed=args.random_seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Input packets: {len(in_df)}")
    print(f"Output flows: {len(out_df)}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
