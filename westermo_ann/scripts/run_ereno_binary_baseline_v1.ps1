# Canonical ERENO binary IDS baseline v1 (streaming, full train_real/test_real).
# Requires: outputs/train_real.csv, outputs/test_real.csv
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Edit these paths to your ERENO comma-separated CSV files:
$TrainCsv = "outputs/train_real.csv"
$TestCsv  = "outputs/test_real.csv"

python src/modeling/train_ereno_stream.py `
  --train-csv $TrainCsv `
  --test-csv $TestCsv `
  --binary-mode `
  --focal-gamma 2 `
  --class-weights off `
  --hidden-dims "256,128" `
  --ga-protect-features "timeFromLastChange,cbStatus,delay,stDiff,SqNum,gooseLen,APDUSize,timestampDiff" `
  --ga-rare-bonus 0.5 `
  --epochs 12 `
  --seed 42 `
  --chunk-rows 200000 `
  --reservoir-rows 120000 `
  --save-model `
  --save-best-pruned `
  --epoch-eval sample `
  --epoch-eval-rows 60000 `
  --output-json reports/binary_baseline_v1.json `
  --artifacts-dir reports/binary_baseline_v1_artifacts
