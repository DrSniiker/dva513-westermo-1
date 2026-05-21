# dva513-westermo-1

Westermo / DVA513: **ERENO IEC-61850 binary IDS** and related ANN experiments.

## Team quick start (ERENO binary IDS)

All Python code and run instructions:

**[`westermo_ann/README.md`](westermo_ann/README.md)**

```powershell
cd westermo_ann
pip install -r requirements.txt
# Set your train/test CSV paths (see README), then:
.\scripts\run_ereno_binary_baseline_v1.cmd
```

Results (local, gitignored): `westermo_ann/reports/binary_baseline_v1_eval.md` and `binary_baseline_v1.json`.

---

## Docker setup (optional — IEC-61850 / UNSW paths)

Each member uses dataset folders from their own machine via `.env`.

### 1) Configure dataset paths

```bash
cp .env.example .env
```

Set:

- `GOOSE_DATASET_PATH` — IEC-61850 preprocessed CSV directory
- `UNSW_DATASET_PATH` — UNSW NB15 archive directory

### 2) Build and run

```bash
docker compose build
docker compose up
```

Default container command runs `unsw_nb15_ann.py`. For the ERENO streaming binary pipeline, run training inside the container with paths from `westermo_ann/README.md`.

### Outputs

- `westermo_ann/outputs` and `westermo_ann/reports` are mounted on the host.
