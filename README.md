# MindMetrix Home Assignment – Biomarker Data Scientist

## Overview

This project implements a reproducible preprocessing and analysis pipeline for physiological time-series data collected during a relaxation exercise. Participants performed **10 trials** of a structured protocol:

- **5 s** baseline
- **30 s** relaxation
- **5 s** break

Multiple physiological signals were recorded per trial. The goal is to engineer meaningful biomarkers, explore inter- and intra-participant structure, and relate those features to participant metadata.

---

## Dataset

The assignment specifies **three** CSV files. Two are named explicitly in the brief; place all raw files in the `data/` directory:

| File | Description |
|---|---|
| `Subjects.csv` | Participant metadata (age, sex, health status, …) |
| `Timeseries.csv` | Raw time-series physiological signals (one row per sample) |
| *(third file – see brief)* | Additional data as provided in the assignment package |

> Assume constant light conditions throughout all recordings.

---

## Project Structure

```
mindmetrix_assignment/
├── data/                   # Raw CSV input files (not tracked in git)
│   ├── Subjects.csv
│   └── Timeseries.csv
├── notebooks/
│   └── analysis.ipynb      # End-to-end analysis notebook
├── src/
│   ├── preprocessing.py    # Signal cleaning and segmentation
│   ├── features.py         # Feature engineering
│   └── visualisation.py    # Reusable plotting helpers
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python ≥ 3.9

### Install dependencies

```bash
pip install -r requirements.txt
```

### Key libraries used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data wrangling and numerical operations |
| `scipy` | Signal filtering (Butterworth, bandpass) and spectral analysis |
| `scikit-learn` | Dimensionality reduction (PCA, UMAP), clustering (K-Means, DBSCAN), scaling |
| `matplotlib` / `seaborn` | Visualisation |
| `neurokit2` *(optional)* | Physiological signal processing (HRV, EDA, respiration) |

Non-standard choices are justified inline in the notebook.

---

## Running the Pipeline

Place the raw CSV files in the `data/` directory, then open the notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells top-to-bottom. The notebook is fully self-contained and reproducible from the raw CSVs.

---

## Pipeline Summary

### 1. Preprocessing
- Load and validate raw data; report missing values and sampling-rate consistency.
- Segment each recording into **baseline**, **relaxation**, and **break** windows per trial.
- Apply appropriate filters per signal type (e.g., Butterworth bandpass for PPG/ECG, low-pass for EDA).
- Normalise signals per participant to remove between-subject amplitude offsets.

### 2. Feature Engineering
Feature extraction is performed separately for each phase and trial. Features are grounded in physiological reasoning:

| Feature family | Physiological rationale |
|---|---|
| Heart-rate variability (HRV) – RMSSD, SDNN, LF/HF ratio | Autonomic nervous system (ANS) tone; relaxation → increased parasympathetic activity |
| EDA tonic level & phasic peaks | Sympathetic arousal; expected to decrease during relaxation |
| Respiration rate & depth | Respiratory sinus arrhythmia couples respiration with HRV |
| Signal power in frequency bands | Objective, decomposable characterisation of oscillatory activity |
| Baseline-normalised delta (relaxation − baseline) | Controls for individual resting differences |

### 3. Unsupervised Analysis
- **PCA** for linear dimensionality reduction and variance explained.
- **UMAP** *(optional)* for non-linear structure discovery.
- **K-Means / hierarchical clustering** to identify participant sub-groups.
- Cluster profiles are related back to metadata (age, sex, health indicators) to assess biomarker sensitivity.

### 4. Biomarker Sensitivity
Sensitivity is assessed by:
- **Effect size** (Cohen's *d*) of features between baseline and relaxation phases.
- **ANOVA / Kruskal-Wallis** across clusters to test whether metadata groups explain feature variance.
- **Intra-class correlation (ICC)** across trials to evaluate within-participant reliability.

---

## Assumptions

- Sampling rate is uniform within each signal channel and consistent across participants.
- Missing samples at trial boundaries are handled by forward-filling up to a short gap threshold; longer gaps exclude the trial from analysis.
- Light conditions are constant, so no luminance correction is applied.

---

## Limitations & Future Work

- With more time: cross-validated predictive modelling (e.g., logistic regression or random-forest) to formally link biomarkers to health metadata.
- Explore adaptive filtering to separate motion artefacts from true physiological variation.
- Implement online feature extraction to support real-time backend integration.

---

## API Integration (Optional)

The preprocessing and feature modules in `src/` are decoupled from the notebook and can be wrapped as a REST microservice:

1. **Ingest endpoint** (`POST /upload`) – accepts raw CSV or streaming sensor data and writes to a time-series store (e.g., InfluxDB, TimescaleDB).
2. **Feature endpoint** (`GET /features/{subject_id}`) – runs the pipeline on demand and returns a JSON payload of engineered features.
3. **Model endpoint** (`GET /cluster/{subject_id}`) – applies a persisted sklearn pipeline and returns cluster assignment + confidence.

Containerising the service with Docker and versioning the sklearn pipeline with MLflow ensures reproducibility as new participant data accumulates.

---

## Time Allocation

| Activity | Estimated time |
|---|---|
| Data exploration & preprocessing | ~1 h |
| Feature engineering | ~1 h |
| Unsupervised analysis & visualisation | ~1 h |
| Report & README | ~30 min |

---

## Author

Assignment submission for the **MindMetrix Biomarker Data Scientist** role.  
Contact: job@mindmetrix.ch (Swiss domain, `.ch`) | Subject: `[Your Name] - Assignment – Biomarker Data Scientist`
