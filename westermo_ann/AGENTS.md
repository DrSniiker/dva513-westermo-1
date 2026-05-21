# AGENTS Constitution - IEC/PowerDuck Project

This file defines how contributors (human and AI) work in this repository.

## Mission Order
1. Keep the pipeline runnable end-to-end.
2. Keep experiments reproducible.
3. Isolate dataset-specific logic (`iec60` vs `powerduck`) to reduce merge conflicts.

## Working Rules
- Use short-lived feature branches and PR into `dev`.
- Do not commit large raw datasets directly to git.
- Keep shared logic in `src/common` and dataset-specific logic in dedicated folders.
- Make small, reversible changes and verify before merging.

## Baseline Commands
- **Binary IDS v1 (full ERENO, streaming):** `scripts/run_ereno_binary_baseline_v1.ps1` → `reports/binary_baseline_v1.json` + `reports/binary_baseline_v1_artifacts/`
- **Binary IDS (in-memory, official test):** `python "src/modeling/train_ereno.py" --use-official-test` → `reports/binary_baseline_eval.md` + `reports/binary_baseline_metrics.json`
- Quick: `--no-ga --seeds 42` or `--max-rows 60000` for subsampled runs
- **Multiclass (optional):** `train_ereno.py --multiclass` → `reports/ereno_multiclass_eval.md`
- **Flow-level:** `python "src/iec60/build_ereno_flow_splits.py"` then `python "src/modeling/train_ereno.py" --representation flow --use-official-test`
- Direct ANN script (advanced): `python "src/modeling/train_ereno_ann.py" --binary-mode --use-official-test`

## Expected Data Locations (ERENO IEC-61850)
- **PCAP → features:** done outside this repo (Wireshark/tshark or dataset release). This codebase starts from **tabular** train/test.
- **Preprocessed CSVs** may live under `IEC-61850/...` or `ereno-iec-61850-ids/` (or any path you pass).
- **Training defaults** expect `train.csv` and `test.csv` at the **repository root**. Publish copies with:
  - `python src/iec60/publish_splits.py` (defaults: `ereno-iec-61850-ids/train.csv` → `train.csv`, same for test), or `--dry-run` to validate only.
- Pipeline description: `src/iec60/README.md`
- Optional UNSW files for legacy experiments: `archive/UNSW_NB15_training-set.csv` and `archive/UNSW_NB15_testing-set.csv`
