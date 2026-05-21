# ERENO IEC-61850 — data path into this repository

This folder holds **dataset-specific** helpers for the ERENO / IEC-61850 track (not PowerDuck).

## End-to-end chain (what happens where)

1. **Raw captures (PCAP)**  
   Produced in the lab or supplied with the dataset. **This repo does not parse PCAP.**  
   Dissection / feature extraction is done with external tools (e.g. Wireshark/tshark, dataset authors’ scripts, or the published preprocessed release).

2. **Preprocessed tabular data (CSV or ARFF)**  
   Row-level (or packet-level) features + labels, as released for ERENO IEC-61850 IDS.  
   You may keep these under `IEC-61850/...`, `ereno-iec-61850-ids/`, or any path on disk.

   **Important:** Some releases store **WEKA ARFF** content in files named `*.csv`.  
   `pandas.read_csv` cannot read those. You must use **comma-separated CSV** (real CSV) for training.  
   If your files start with `@relation`, export from WEKA as CSV or convert ARFF → CSV first; `publish_splits.py` will **refuse** to copy ARFF-as-`.csv` sources.

3. **Publish into the repo root (row-level defaults for training)**  
   Modeling defaults expect **`train.csv`** and **`test.csv`** at the repository root.

   ```bash
   python src/iec60/publish_splits.py
   ```

   Or explicit sources / `--dry-run` — see `publish_splits.py --help`.

4. **Flow aggregation (IEC GOOSE–oriented, Python — *first-class*)**  
   Implementation: `flow_aggregate.py` (library), invoked by:

   - **Batch train + test → flow CSVs** (recommended before `--representation flow`):

     ```bash
     python src/iec60/build_ereno_flow_splits.py
     ```

     Defaults: read repo-root `train.csv` / `test.csv`, write `outputs/train_flows.csv` and `outputs/test_flows.csv`.

   - **Single file** (same logic):

     ```bash
     python src/modeling/ereno_to_flows.py --input-csv path/to/rows.csv --output-csv path/to/flows.csv
     ```

5. **Training / evaluation**  

   - **Row-level (wide IEC rows):**

     ```bash
     python src/modeling/train_ereno.py --representation row --use-official-test
     ```

     (`row` is the default; you may omit `--representation row`.)

   - **Flow-level:**

     ```bash
     python src/modeling/train_ereno.py --representation flow --use-official-test
     ```

     Requires step 4 outputs (or pass custom `--train-csv` / `--test-csv` to your flow files).

6. **Preprocessing inside the ANN trainer** (`src/iec60/preprocess_ereno.py`, applied in `train_ereno_ann.py` before encoding):

   - **Identifier columns** removed (`id`, `flow_id`, …).
   - **Exact duplicate rows** removed within each split (fit / eval separately). Disable with `--no-drop-duplicates`.
   - **Numeric variance filter** on the **fit** split: drop columns with variance ≤ `--variance-threshold` (default `0` removes constants only). Same columns dropped from eval. Disable with `--skip-variance-filter`.
   - Then: missing-value handling, scaling, categorical encoding, class weights (existing `encode_features` path).

7. **GA feature selection**  
   - In-memory (subsample): `train_ereno.py` enables GA by default unless **`--no-ga`**.  
   - Full dataset (~3M rows): `train_ereno_stream.py` with **`--binary-mode`** (see `scripts/run_ereno_binary_baseline_v1.ps1`).  
   Direct `train_ereno_ann.py` still requires `--use-ga-feature-selection` explicitly if you bypass the entry scripts.

8. **C `flow_dpi/`**  
   Separate IP/TCP/UDP metering code — **not** wired into the Python ANN path. See `src/flow_dpi/README.md`.

## Layout constants

See `src/iec60/paths.py` for `REPO_ROOT`, IEC-61850 preprocessed paths, flow output paths, etc.

## ARFF

If you only have **`.arff`** files, convert them to CSV outside this script (e.g. WEKA export, or a one-off Python/SciPy job) so that `publish_splits.py` receives CSV inputs. Full ARFF loads are often not memory-safe for multi-gigabyte files inside a single `read_arff` call.
