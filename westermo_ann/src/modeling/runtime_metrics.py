"""Standard classification + runtime metrics for every experiment export."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from time import perf_counter

# Flat keys written to JSON aggregates, per-seed rows, and markdown tables.
STANDARD_FLAT_RUNTIME_FIELDS = (
    "total_params",
    "estimated_size_mb_fp32",
    "inference_ms_per_sample",
    "inference_ms_per_sample_batch1",
    "cpu_rss_peak_mb",
    "cpu_percent_peak",
    "cpu_percent_single_core_peak",
)

STANDARD_CLASSIFICATION_FIELDS = ("macro_f1", "mcc", "accuracy")


def model_size_stats(model: nn.Module) -> dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    return {
        "total_params": int(params),
        "estimated_size_mb_fp32": float((params * 4) / (1024**2)),
    }


def _logical_cpu_count() -> int:
    try:
        import psutil

        return max(1, int(psutil.cpu_count(logical=True) or 1))
    except Exception:
        return 1


def _sample_process_resources(proc: Any) -> tuple[float | None, float | None, float | None]:
    """Return (rss_mb, cpu_percent_peak_since_last, None) — caller tracks peak CPU."""
    if proc is None:
        return None, None, None
    rss_mb = float(proc.memory_info().rss / (1024**2))
    cpu_pct = float(proc.cpu_percent(None))
    return rss_mb, cpu_pct, None


def latency_benchmark(
    model: nn.Module,
    x_eval: torch.Tensor,
    runs: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    """Per-batch-size latency; process RSS + CPU % peaks during timed inference (psutil)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    proc = psutil.Process() if psutil is not None else None
    n_cpu = _logical_cpu_count()

    if proc is not None:
        proc.cpu_percent(None)

    model.eval()
    out: dict[str, Any] = {}
    rss_peak_mb: float | None = None
    cpu_peak: float = 0.0

    dev = x_eval.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    for bs in (1, 8, 32):
        if x_eval.size(0) < bs:
            continue
        sample = x_eval[:bs]
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample)
                if proc is not None:
                    rss, cpu, _ = _sample_process_resources(proc)
                    if rss is not None:
                        rss_peak_mb = max(rss_peak_mb or 0.0, rss)
                    if cpu is not None:
                        cpu_peak = max(cpu_peak, cpu)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
        t0 = perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(sample)
                if proc is not None:
                    rss, cpu, _ = _sample_process_resources(proc)
                    if rss is not None:
                        rss_peak_mb = max(rss_peak_mb or 0.0, rss)
                    if cpu is not None:
                        cpu_peak = max(cpu_peak, cpu)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
        ms = ((perf_counter() - t0) / runs) * 1000.0
        out[f"batch_{bs}"] = {
            "mean_ms_per_batch": float(ms),
            "mean_ms_per_sample": float(ms / bs),
            "runs": int(runs),
            "warmup_runs": int(warmup),
        }

    if proc is not None and rss_peak_mb is not None:
        out["process_rss_peak_mb"] = float(rss_peak_mb)
    if cpu_peak > 0.0:
        out["cpu_percent_peak"] = float(cpu_peak)
        out["cpu_percent_single_core_peak"] = float(min(100.0, cpu_peak / n_cpu))
        out["logical_cpu_count"] = int(n_cpu)
    if dev.type == "cuda":
        out["gpu_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(dev) / (1024**2))
    return out


def _canonical_ms_per_sample(latency: dict[str, Any]) -> float | None:
    for key in ("batch_32", "batch_8", "batch_1"):
        block = latency.get(key)
        if isinstance(block, dict) and "mean_ms_per_sample" in block:
            return float(block["mean_ms_per_sample"])
    return None


def _batch1_ms_per_sample(latency: dict[str, Any]) -> float | None:
    block = latency.get("batch_1")
    if isinstance(block, dict) and "mean_ms_per_sample" in block:
        return float(block["mean_ms_per_sample"])
    return None


def _split_latency_benchmark_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    batches: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("batch_") and isinstance(v, dict):
            batches[k] = v
        elif not k.startswith("batch_"):
            extra[k] = v
    return batches, extra


def inference_evaluation(
    model: nn.Module,
    x_eval: torch.Tensor,
    device: torch.device,
    runs: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    """Latency by batch size, RSS peak, CPU % peak (process), optional GPU peak."""
    raw = latency_benchmark(model, x_eval, runs=runs, warmup=warmup)
    by_bs, extra = _split_latency_benchmark_payload(raw)
    mem: dict[str, Any] = {
        "cpu_rss_mb_peak_during_inference": extra.get("process_rss_peak_mb"),
        "cpu_percent_peak_during_inference": extra.get("cpu_percent_peak"),
        "cpu_percent_single_core_peak_during_inference": extra.get("cpu_percent_single_core_peak"),
        "logical_cpu_count": extra.get("logical_cpu_count"),
        "gpu_peak_reserved_mb": extra.get("gpu_peak_reserved_mb"),
        "notes": [],
    }
    if mem["cpu_rss_mb_peak_during_inference"] is None:
        mem["notes"].append("CPU RSS peak unavailable (install psutil).")
    if mem["cpu_percent_peak_during_inference"] is None:
        mem["notes"].append("CPU % peak unavailable (install psutil).")
    if device.type != "cuda":
        mem["notes"].append("GPU peak reserved: N/A (CPU-only run).")
    return {
        "latency_by_batch_size": by_bs,
        "memory": mem,
        "canonical_inference_ms_per_sample": _canonical_ms_per_sample(raw),
        "inference_ms_per_sample_batch1": _batch1_ms_per_sample(raw),
    }


def runtime_block(model_stats: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    """Canonical runtime section for experiment JSON (baseline / prune / per-seed)."""
    mem = inference.get("memory") or {}
    return {
        "total_params": int(model_stats["total_params"]),
        "estimated_size_mb_fp32": float(model_stats["estimated_size_mb_fp32"]),
        "inference_ms_per_sample": inference.get("canonical_inference_ms_per_sample"),
        "inference_ms_per_sample_batch1": inference.get("inference_ms_per_sample_batch1"),
        "cpu_rss_peak_mb": mem.get("cpu_rss_mb_peak_during_inference"),
        "cpu_percent_peak": mem.get("cpu_percent_peak_during_inference"),
        "cpu_percent_single_core_peak": mem.get("cpu_percent_single_core_peak_during_inference"),
        "logical_cpu_count": mem.get("logical_cpu_count"),
        "gpu_peak_reserved_mb": mem.get("gpu_peak_reserved_mb"),
        "latency_by_batch_size": inference.get("latency_by_batch_size"),
        "notes": list(mem.get("notes") or []),
    }


def measure_model_runtime(
    model: nn.Module,
    x_eval: torch.Tensor,
    device: torch.device,
    *,
    runs: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    """Model size + inference benchmark in one call."""
    stats = model_size_stats(model)
    inf = inference_evaluation(model, x_eval, device, runs=runs, warmup=warmup)
    return {
        "model_stats": stats,
        "inference": inf,
        "runtime": runtime_block(stats, inf),
    }


def flat_runtime_from_block(runtime: dict[str, Any]) -> dict[str, Any]:
    """Top-level flat fields for multi-seed tables and backward-compatible JSON."""
    return {k: runtime.get(k) for k in STANDARD_FLAT_RUNTIME_FIELDS}


def per_seed_result_row(
    *,
    seed: int,
    metrics: dict[str, Any],
    runtime: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": int(seed),
        "metrics": metrics,
        "runtime": runtime,
        **flat_runtime_from_block(runtime),
        # Legacy aliases used by older markdown/aggregate code paths.
        "inference_ms_per_sample": runtime.get("inference_ms_per_sample"),
        "cpu_rss_peak_mb": runtime.get("cpu_rss_peak_mb"),
        "gpu_peak_reserved_mb": runtime.get("gpu_peak_reserved_mb"),
        "model_size_mb_fp32": runtime.get("estimated_size_mb_fp32"),
        "total_params": runtime.get("total_params"),
        "cpu_percent_peak": runtime.get("cpu_percent_peak"),
        "cpu_percent_single_core_peak": runtime.get("cpu_percent_single_core_peak"),
    }
    if extra:
        row.update(extra)
    return row


def baseline_result_bundle(
    metrics: dict[str, Any],
    model: nn.Module,
    x_eval: torch.Tensor,
    device: torch.device,
    *,
    runs: int = 200,
    warmup: int = 20,
    sparsity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard baseline block for train_ereno_ann / stream exports."""
    measured = measure_model_runtime(model, x_eval, device, runs=runs, warmup=warmup)
    out: dict[str, Any] = {
        "metrics": metrics,
        "model_stats": measured["model_stats"],
        "inference": measured["inference"],
        "runtime": measured["runtime"],
    }
    if sparsity is not None:
        out["sparsity"] = sparsity
    return out


def _fmt_opt(value: Any, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_standard_summary_md(
    path: Any,
    *,
    title: str,
    metrics: dict[str, Any],
    runtime: dict[str, Any] | None,
    subtitle: str = "",
) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if subtitle:
        lines.extend([subtitle, ""])
    lines.extend(
        [
            "## Classification (test set)",
            "",
            "| Macro F1 | MCC | Accuracy | Weighted F1 | Balanced Acc |",
            "|---:|---:|---:|---:|---:|",
            "| "
            f"{_fmt_opt(metrics.get('macro_f1'))} | {_fmt_opt(metrics.get('mcc'))} | "
            f"{_fmt_opt(metrics.get('accuracy'))} | {_fmt_opt(metrics.get('weighted_f1'))} | "
            f"{_fmt_opt(metrics.get('balanced_accuracy'))} |",
            "",
        ]
    )
    if runtime is not None:
        lines.extend(
            [
                "## Runtime (edge-relevant)",
                "",
                "| Params | Model MB (FP32) | Infer ms/sample | Infer ms/sample (batch=1) | RSS peak MB | CPU % peak | CPU % 1-core peak |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                "| "
                f"{_fmt_opt(runtime.get('total_params'), digits=0)} | "
                f"{_fmt_opt(runtime.get('estimated_size_mb_fp32'), digits=3)} | "
                f"{_fmt_opt(runtime.get('inference_ms_per_sample'))} | "
                f"{_fmt_opt(runtime.get('inference_ms_per_sample_batch1'))} | "
                f"{_fmt_opt(runtime.get('cpu_rss_peak_mb'), digits=1)} | "
                f"{_fmt_opt(runtime.get('cpu_percent_peak'), digits=1)} | "
                f"{_fmt_opt(runtime.get('cpu_percent_single_core_peak'), digits=1)} |",
                "",
            ]
        )
        notes = runtime.get("notes") or []
        if notes:
            lines.append("Notes: " + " ".join(notes))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_multi_seed_summary_md(
    path: Any,
    *,
    title: str,
    seeds: list[int],
    aggregate: dict[str, dict[str, float]],
    per_seed_rows: list[dict[str, Any]],
) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Mean ± std (classification)",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("macro_f1", "Macro F1"),
        ("mcc", "MCC"),
        ("accuracy", "Accuracy"),
        ("weighted_f1", "Weighted F1"),
        ("balanced_accuracy", "Balanced accuracy"),
    ]:
        if key in aggregate:
            m = aggregate[key]
            lines.append(f"| {label} | {m['mean']:.4f} | {m['std']:.4f} |")

    runtime_agg_keys = [
        ("total_params", "Parameter count"),
        ("estimated_size_mb_fp32", "Model MB (FP32 est.)"),
        ("inference_ms_per_sample", "Inference ms/sample (canonical)"),
        ("inference_ms_per_sample_batch1", "Inference ms/sample (batch=1)"),
        ("cpu_rss_peak_mb", "CPU RSS peak (MB)"),
        ("cpu_percent_peak", "CPU % peak (process)"),
        ("cpu_percent_single_core_peak", "CPU % peak (1-core equiv.)"),
        ("gpu_peak_reserved_mb", "GPU peak reserved (MB)"),
    ]
    if any(k in aggregate for k, _ in runtime_agg_keys):
        lines.extend(["", "## Mean ± std (runtime)", "", "| Metric | Mean | Std |", "|---|---:|---:|"])
        for key, label in runtime_agg_keys:
            if key not in aggregate:
                continue
            m = aggregate[key]
            lines.append(f"| {label} | {m['mean']:.4f} | {m['std']:.4f} |")

    lines.extend(
        [
            "",
            "## Per seed",
            "",
            "| Seed | Macro F1 | MCC | Acc | Params | Model MB | ms/sample | ms/sample b=1 | RSS MB | CPU % | CPU % 1-core |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed_rows:
        met = row["metrics"]
        rt = row.get("runtime") or {}
        lines.append(
            f"| {row['seed']} | {met['macro_f1']:.4f} | {met['mcc']:.4f} | {met['accuracy']:.4f} | "
            f"{_fmt_opt(rt.get('total_params'), digits=0)} | {_fmt_opt(rt.get('estimated_size_mb_fp32'), digits=2)} | "
            f"{_fmt_opt(rt.get('inference_ms_per_sample'))} | {_fmt_opt(rt.get('inference_ms_per_sample_batch1'))} | "
            f"{_fmt_opt(rt.get('cpu_rss_peak_mb'), digits=1)} | "
            f"{_fmt_opt(rt.get('cpu_percent_peak'), digits=1)} | "
            f"{_fmt_opt(rt.get('cpu_percent_single_core_peak'), digits=1)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
