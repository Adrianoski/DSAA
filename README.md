# Changepoint Detection for Satellite Telemetry Anomaly Detection

Label-free changepoint detection (CPD) on multivariate satellite telemetry,
with two variants:

1. **Unsupervised** — STFT features → penalty sweep (PELT) → per-dimension gain
   elbow → auto-calibrated **gain gate** (parameter-free *knee* of the training
   gain distribution).
2. **Classifier** — the same pipeline plus a per-changepoint **Random Forest**
   re-ranker (the only component using annotations: semi-supervised).

`run_clf_knee.sh` produces **both** views at the same parameter-free threshold:
CPD-only (raw gate output) and CPD+RandomForest.

## Requirements

Python 3.10 with: `numpy`, `pandas`, `scipy`, `ruptures`, `scikit-learn`,
`matplotlib`. The code is self-contained (no external local modules).

## Data

This repo does **not** ship data (see `.gitignore`). Provide the ESA-ADB
Mission 1 data and point the pipeline to it via an environment variable:

```bash
export ESA_ADB_DATA=/path/to/ESA-ADB/data
```
Expected layout under `$ESA_ADB_DATA`:
```
ESA-Mission1/labels.csv
ESA-Mission1/anomaly_types.csv
preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.{train,test}.csv
```
The pipeline also expects the per-channel standardised streams
`84_months.{train,test}.unsup_std.csv` in the working directory (produced by the
standardisation step of the unsupervised variant).

## How to run

### Unsupervised variant
```bash
python3 -u pen_sweep_unsupervised.py
```
Rebuilds the standardised CSVs, runs CPD with the auto-calibrated gain gate, and
evaluates. Output → `cpd_output_unsupervised/`.

### Classifier variant (parameter-free, knee gate)
```bash
# 1) calibrate the gain gate (knee of the training gain distribution)
python3 -u calibrate_gain.py
python3 -c "import json;print('knee =',json.load(open('gain_distribution_summary.json'))['knee']['knee_gain'])"

# 2) full pipeline: build RF training set -> train -> test -> evaluate
./run_clf_knee.sh
```
`run_clf_knee.sh` reads the knee from `gain_distribution_summary.json`
automatically. Output → `cpd_output_classified_knee/` + per-variant eval logs.

## Key configuration (`config.py`)

- `WINDOW_DAYS = 5` — fixed non-overlapping windows.
- `CPD_MIN_SIZE`, `CPD_JUMP` — PELT segment constraints.
- STFT band / `nperseg` / overlap.
- `PER_DIM_MIN_CONTRIBUTING` — conservative per-dimension elbow filter.

## Notes

- One CPD pass over the training windows is the dominant cost; the classifier
  pipeline runs two heavy passes (training-set + test-set).
- Changing `CPD_MIN_SIZE` invalidates the gate calibration and requires
  re-running both calibration and pipeline.
