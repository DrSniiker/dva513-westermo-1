# dva513-westermo-1

Westermo project course ANN network pruning.

## Docker setup

This repository includes a Docker setup so each group member can run the same code while using dataset folders from their own machine.

### 1) Configure dataset paths

Copy `.env.example` to `.env` and set absolute dataset paths:

```bash
cp .env.example .env
```

Set:
- `GOOSE_DATASET_PATH`: directory containing the IEC-61850 preprocessed CSV files.
- `UNSW_DATASET_PATH`: directory containing `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.

### 2) Build image

```bash
docker compose build
```

### 3) Run training

Default command runs:

```bash
python unsw_nb15_ann.py
```

Start it with:

```bash
docker compose up
```

### 4) Run the multiclass UNSW pipeline instead

```bash
docker compose run --rm model-trainer python src/modeling/train_multiclass.py
```

## Outputs

- `westermo_ann/outputs` is mounted from the host.
- `westermo_ann/reports` is mounted from the host.

Generated files stay on your machine and are not lost when the container exits.

## Notes

- Dataset paths are passed through volume mounts and environment variables, so teammates can use different local paths without changing code.
- If you need GPU support later, we can add a CUDA-enabled image and runtime settings.
