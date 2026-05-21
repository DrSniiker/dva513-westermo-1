# ERENO IEC-61850 binary IDS — team run guide

Binary **normal vs attack** detection on ERENO tabular data.  
**Main trainer:** `src/modeling/train_ereno_stream.py` (full dataset, chunked — millions of rows).

Work in this folder: **`westermo_ann`** (paths below assume you are here).

---

## 1. Install (once per machine)

```powershell
cd path\to\repo\westermo_ann
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires **Python 3.11+** and enough disk/RAM for your CSV files.

---

## 2. Point to your ERENO dataset

Training needs **real comma-separated CSV** files with a **`class`** column (or WEKA ARFF you convert first).

### Option A — Pass paths directly (recommended)

Use **absolute or relative paths** to your train and test files:

```powershell
python src/modeling/train_ereno_stream.py `
  --train-csv "D:\datasets\ereno\train.csv" `
  --test-csv "D:\datasets\ereno\test.csv" `
  --binary-mode `
  --output-json reports/my_run.json `
  --save-model
```

On Linux/macOS, use forward slashes: `--train-csv /data/ereno/train.csv`.

### Option B — Standard layout under this project

1. Put files anywhere, e.g. `ereno-iec-61850-ids\train.csv` and `test.csv`.
2. If they are **ARFF** (file starts with `@relation`), convert:

```powershell
python src/iec60/weka_arff_to_csv.py --input "path\to\train.arff" --output outputs/train_real.csv
python src/iec60/weka_arff_to_csv.py --input "path\to\test.arff" --output outputs/test_real.csv
```

3. Or copy valid CSVs:

```powershell
python src/iec60/publish_splits.py `
  --train-source "path\to\your\train.csv" `
  --test-source "path\to\your\test.csv" `
  --dest-dir outputs
```

Then use `--train-csv outputs/train.csv` and `--test-csv outputs/test.csv` (or keep `outputs/train_real.csv` names).

**Do not** use root `train.csv` / `test.csv` unless `Get-Content train.csv -TotalCount 1` shows a normal header — not `@relation`.

---

## 3. Run the canonical binary baseline

Same settings as the locked v1 experiment (12 epochs, focal loss, GA, pruning sweep):

```powershell
# Windows — works even if .ps1 scripts are blocked:
.\scripts\run_ereno_binary_baseline_v1.cmd
```

Edit `scripts\run_ereno_binary_baseline_v1.cmd` (or `.ps1`) and change the two lines:

```bat
  --train-csv outputs/train_real.csv ^
  --test-csv outputs/test_real.csv ^
```

to your paths, **or** run Python explicitly:

```powershell
python src/modeling/train_ereno_stream.py `
  --train-csv "YOUR\train.csv" `
  --test-csv "YOUR\test.csv" `
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
```

**Always include `--binary-mode`** for normal vs attack. Without it, the trainer uses multiclass labels (different task).

---

## 4. What to open after the run

| File | Purpose |
|------|---------|
| **`reports/binary_baseline_v1_eval.md`** | **Share with the team** — attack F1, false alarms (FPR), missed attacks (FNR), confusion matrix, pruning table |
| `reports/binary_baseline_v1.json` | Full metrics, `epoch_eval_history`, `pruning_sweep` |
| `reports/binary_baseline_v1_seed42.pt` | Trained model (dense weights) |
| `reports/binary_baseline_v1_pruned*_seed42.pt` | Best pruned model (if `--save-best-pruned`) |
| `reports/binary_baseline_v1_artifacts/` | ROC, PR, confusion matrix plots |

The terminal ends with a **STREAM TRAINING FINISHED** box listing these paths.

### Per-epoch confusion matrices

During training you will see lines like:

```text
[stream] epoch 01/12 confusion_matrix (rows=true, cols=pred):
```

That is a **60k stratified snapshot** from the test file for monitoring — not the final full-test score. Final numbers are after all 12 epochs on the **full test CSV**.

---

## 5. Faster smoke test

```powershell
python src/modeling/train_ereno_stream.py `
  --train-csv "YOUR\train.csv" `
  --test-csv "YOUR\test.csv" `
  --binary-mode `
  --epochs 3 `
  --no-ga `
  --epoch-eval off `
  --pruning-ratios "" `
  --output-json reports/smoke.json
```

For small in-memory CSVs only: `python src/modeling/train_ereno.py --train-csv ... --test-csv ... --binary-mode --epochs 3 --no-ga`.

---

## 6. Repository layout

```
westermo_ann/
├── scripts/run_ereno_binary_baseline_v1.cmd   # one-click baseline (edit CSV paths inside)
├── src/modeling/train_ereno_stream.py       # main entry
├── src/iec60/weka_arff_to_csv.py            # ARFF → CSV
├── src/iec60/publish_splits.py              # copy CSVs to a folder
├── requirements.txt
└── reports/                                 # created locally (gitignored)
```

More detail: [`src/iec60/README.md`](src/iec60/README.md), [`AGENTS.md`](AGENTS.md).

---

## 7. Protected GA features (for reports)

`--ga-protect-features` keeps listed IEC/GOOSE columns **always included** in GA feature selection (domain prior, not test-set leakage). Default list is in the script above.

---

## Dependencies

`pip install -r requirements.txt` — PyTorch, pandas, scikit-learn, matplotlib, psutil.
