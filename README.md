# MindMetrix Technical Challenge

This repository contains a physiological data analysis pipeline built for the MindMetrix technical challenge. It processes multimodal biosignals — heart rate (PPG), pupil diameter, gaze, and motion — recorded from participants across baseline and relaxation phases. The pipeline covers the full workflow from raw data cleaning and feature engineering through to statistical biomarker analysis, PCA-based dimensionality reduction, and unsupervised clustering to identify participant response profiles.

## Setup

Create and activate a conda environment:

```bash
conda create -n mindmetrix python=3.11
conda activate mindmetrix
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

The `data/` directory is not included in this repository. Place the following files there before running the pipeline:

```
data/
├── timeseries.csv   # Raw physiological time-series
└── subjects.csv     # Participant metadata
```

## Running the pipeline

Launch Jupyter and open `analysis.ipynb`:

```bash
jupyter notebook analysis.ipynb
```

Run all cells top to bottom (**Kernel → Restart & Run All**). The notebook executes the full pipeline in order:

1. **Preprocessing** — loads `data/timeseries.csv` and `data/subjects.csv`, removes duplicates, cleans physiological outliers and tracking artifacts.
2. **Feature extraction** — computes per-phase signal statistics, relaxation deltas, and cross-trial aggregates; produces one row per subject.
3. **Biomarker sensitivity** — Wilcoxon signed-rank tests + Cohen's d for each signal across phases.
4. **Metadata correlations** — Spearman correlations with STAI scores; Mann-Whitney U tests for binary demographic groups.
5. **PCA** — dimensionality reduction with explained variance and component loadings.
6. **Clustering** — K-Means with elbow and silhouette selection; cluster profiling against metadata.

All outputs (tables, plots) are rendered inline. No configuration is required — data paths are set at the top of the notebook.

## Project structure

```
technical_challenge/
├── data/
│   ├── timeseries.csv     # Raw physiological time-series
│   └── subjects.csv       # Participant metadata
├── assignment/            # Assignment brief and original files
├── src/
│   ├── preprocessing.py   # Data loading and cleaning
│   ├── features.py        # Signal transforms and feature extraction
│   └── analysis.py        # Statistical tests, PCA, and clustering
├── requirements.txt
├── eda.ipynb              # Exploratory data analysis
└── analysis.ipynb         # Main notebook — run this
```

## Backend API integration

The notebook pipeline is designed as a set of pure, stateless functions, which makes it straightforward to wrap in a production API.

**Offline training (once).** After running the notebook on the reference dataset, the fitted `StandardScaler`, `PCA`, and `KMeans` objects are serialised to disk with `joblib`. These artefacts encode the population-level feature distribution and cluster boundaries learned from the training cohort.

**Inference endpoint.** A FastAPI service exposes a single `POST /session` endpoint. When a relaxation session ends, the wearable device (or a session-management service) sends the raw timeseries and subject metadata to this endpoint. The handler calls `preprocess()` followed by `build_feature_matrix()` to produce the same feature representation used during training, then loads the serialised scaler/PCA/K-Means to assign the subject to a relaxation profile cluster. The response returns the cluster label and key biomarker values (e.g. HR delta, pupil delta) for display in a clinician dashboard or mobile app.


**Scaling.** For higher session throughput, the feature extraction step can be offloaded to a task queue (e.g. Celery with Redis) while the API layer remains lightweight and stateless, returning a job ID immediately and delivering results asynchronously.
