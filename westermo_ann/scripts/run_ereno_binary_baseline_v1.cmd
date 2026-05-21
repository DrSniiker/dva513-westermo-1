@echo off
REM Binary baseline v1. Edit TRAIN_CSV and TEST_CSV below to your ERENO comma-separated CSV paths.
REM Works when PowerShell blocks .ps1 scripts.
cd /d "%~dp0.."
set TRAIN_CSV=outputs\train_real.csv
set TEST_CSV=outputs\test_real.csv
python src/modeling/train_ereno_stream.py ^
  --train-csv %TRAIN_CSV% ^
  --test-csv %TEST_CSV% ^
  --binary-mode ^
  --focal-gamma 2 ^
  --class-weights off ^
  --hidden-dims 256,128 ^
  --ga-protect-features timeFromLastChange,cbStatus,delay,stDiff,SqNum,gooseLen,APDUSize,timestampDiff ^
  --ga-rare-bonus 0.5 ^
  --epochs 12 ^
  --seed 42 ^
  --chunk-rows 200000 ^
  --reservoir-rows 120000 ^
  --save-model ^
  --save-best-pruned ^
  --epoch-eval sample ^
  --epoch-eval-rows 60000 ^
  --output-json reports/binary_baseline_v1.json ^
  --artifacts-dir reports/binary_baseline_v1_artifacts
if errorlevel 1 exit /b 1
