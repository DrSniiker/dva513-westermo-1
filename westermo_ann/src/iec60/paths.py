from __future__ import annotations

from pathlib import Path

# Repository root (parent of `src/`)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default layout when the full IEC-61850 tree is present (see README)
IEC61850_PREPROCESSED_DIR = (
    REPO_ROOT / "IEC-61850" / "preprocessed csv files" / "preprocessed csv files"
)

# Alternate copy sometimes shipped alongside the repo (often ARFF mislabeled as *.csv — check before publish)
ERENO_PACKAGE_DIR = REPO_ROOT / "ereno-iec-61850-ids"

DEFAULT_PUBLISH_TRAIN = ERENO_PACKAGE_DIR / "train.csv"
DEFAULT_PUBLISH_TEST = ERENO_PACKAGE_DIR / "test.csv"

REPO_TRAIN_CSV = REPO_ROOT / "train.csv"
REPO_TEST_CSV = REPO_ROOT / "test.csv"

# Flow-level tables built from row CSVs (`build_ereno_flow_splits.py`)
FLOW_OUTPUT_DIR = REPO_ROOT / "outputs"
FLOW_TRAIN_CSV = FLOW_OUTPUT_DIR / "train_flows.csv"
FLOW_TEST_CSV = FLOW_OUTPUT_DIR / "test_flows.csv"

# Comma-separated row CSVs for streaming trainer (convert ARFF with `weka_arff_to_csv.py` first)
STREAM_TRAIN_CSV = FLOW_OUTPUT_DIR / "train_real.csv"
STREAM_TEST_CSV = FLOW_OUTPUT_DIR / "test_real.csv"

# Canonical binary baseline v1 artifacts (regenerate with `scripts/run_ereno_binary_baseline_v1.ps1`)
BINARY_BASELINE_JSON = REPO_ROOT / "reports" / "binary_baseline_v1.json"
BINARY_BASELINE_ARTIFACTS_DIR = REPO_ROOT / "reports" / "binary_baseline_v1_artifacts"
