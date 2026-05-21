from __future__ import annotations

"""Entry point for IEC-61850 IDS: binary normal vs attack by default; optional multiclass."""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modeling.train_ereno_ann import build_parser, run_from_parsed_args


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = BASE_DIR / "train.csv"
DEFAULT_TEST_CSV = BASE_DIR / "test.csv"
FLOW_TRAIN_CSV = BASE_DIR / "outputs" / "train_flows.csv"
FLOW_TEST_CSV = BASE_DIR / "outputs" / "test_flows.csv"
DEFAULT_TARGET_COL = "class"
RANDOM_STATE = 42
# Thesis-style baseline: three seeds, mean ± std in reports (override with e.g. --seeds 42 for a quick run).
DEFAULT_SEEDS = "41,42,43"

OUTPUT_DIR = BASE_DIR / "reports"
DEFAULT_METRICS_JSON = OUTPUT_DIR / "binary_baseline_metrics.json"
DEFAULT_SUMMARY_MD = OUTPUT_DIR / "binary_baseline_eval.md"
MULTICLASS_METRICS_JSON = OUTPUT_DIR / "ereno_multiclass_metrics.json"
MULTICLASS_SUMMARY_MD = OUTPUT_DIR / "ereno_multiclass_eval.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--representation",
        choices=("row", "flow"),
        default="row",
        help="row=packet/event CSVs at repo root (train.csv/test.csv). "
        "flow=flow-feature CSVs (default outputs/train_flows.csv, test_flows.csv); build via "
        "src/iec60/build_ereno_flow_splits.py first.",
    )
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--use-official-test", action="store_true")
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help=f"Comma-separated RNG seeds (default {DEFAULT_SEEDS!r}). Use empty string for a single run with seed {RANDOM_STATE}.",
    )
    parser.add_argument(
        "--no-drop-duplicates",
        action="store_true",
        help="Keep exact duplicate rows (default: drop duplicates within each split).",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.0,
        help="Drop numeric columns with variance <= value on fit split (default 0 = constants only).",
    )
    parser.add_argument("--skip-variance-filter", action="store_true", help="Disable variance-based column filter.")
    parser.add_argument(
        "--no-ga",
        action="store_true",
        help="Disable genetic-algorithm feature selection (faster; uses all encoded features).",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=16,
        help="GA population size (default tuned for thesis runs).",
    )
    parser.add_argument("--ga-generations", type=int, default=8, help="GA generations (default thesis).")
    parser.add_argument("--ga-eval-epochs", type=int, default=3, help="Training epochs per GA fitness eval.")
    parser.add_argument(
        "--ga-sample-size",
        type=int,
        default=15000,
        help="Max train rows used inside GA (0 = all rows; large data: keep default to cap runtime).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward to ANN trainer: numbered pipeline logs, every epoch, per-class report.",
    )
    parser.add_argument("--save-model", action="store_true", help="Save trained baseline weights next to metrics JSON.")
    parser.add_argument(
        "--pruning-ratios",
        type=str,
        default="",
        help="Optional comma-separated global L1-unstructured prune ratios in [0,1), e.g. 0.2,0.5. "
        "Applied after baseline training (same as train_ereno_ann.py --pruning-ratios). Empty = no pruning sweep.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Forward to ANN trainer: cap rows per split after load (0 = all).",
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="Train on original attack classes (default is binary: normal vs attack).",
    )
    return parser.parse_args()


def main() -> None:
    mc = parse_args()
    if mc.multiclass:
        if mc.metrics_json == DEFAULT_METRICS_JSON:
            mc.metrics_json = MULTICLASS_METRICS_JSON
        if mc.summary_md == DEFAULT_SUMMARY_MD:
            mc.summary_md = MULTICLASS_SUMMARY_MD
    train_csv = mc.train_csv
    test_csv = mc.test_csv
    if mc.representation == "flow":
        if train_csv == DEFAULT_TRAIN_CSV:
            train_csv = FLOW_TRAIN_CSV
        if test_csv == DEFAULT_TEST_CSV:
            test_csv = FLOW_TEST_CSV
        if not train_csv.is_file() or not test_csv.is_file():
            raise FileNotFoundError(
                "Flow representation selected but flow CSVs are missing.\n"
                "Build from row-level train.csv / test.csv:\n"
                "  python src/iec60/build_ereno_flow_splits.py\n"
                f"Expected:\n  {train_csv}\n  {test_csv}"
            )
    argv = [
        "--train-csv",
        str(train_csv),
        "--test-csv",
        str(test_csv),
        "--target-col",
        mc.target_col,
        "--output-json",
        str(mc.metrics_json),
        "--summary-md",
        str(mc.summary_md),
        "--pruning-ratios",
        str(mc.pruning_ratios),
    ]
    seeds_arg = (mc.seeds or "").strip()
    if seeds_arg:
        argv.extend(["--seeds", seeds_arg])
    else:
        argv.extend(["--seed", str(RANDOM_STATE)])
    if mc.use_official_test:
        argv.append("--use-official-test")
    if int(mc.max_rows) > 0:
        argv.extend(["--max-rows", str(int(mc.max_rows))])
    if mc.no_drop_duplicates:
        argv.append("--no-drop-duplicates")
    if mc.skip_variance_filter:
        argv.append("--skip-variance-filter")
    else:
        argv.extend(["--variance-threshold", str(mc.variance_threshold)])
    if not mc.no_ga:
        argv.append("--use-ga-feature-selection")
        argv.extend(
            [
                "--ga-population",
                str(mc.ga_population),
                "--ga-generations",
                str(mc.ga_generations),
                "--ga-eval-epochs",
                str(mc.ga_eval_epochs),
                "--ga-sample-size",
                str(mc.ga_sample_size),
            ]
        )
    if mc.verbose:
        argv.append("--verbose")
    if mc.save_model:
        argv.append("--save-model")
    if not mc.multiclass:
        argv.append("--binary-mode")
    ann_args = build_parser().parse_args(argv)
    run_from_parsed_args(ann_args)


if __name__ == "__main__":
    main()
