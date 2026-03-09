"""
preprocessing.py
----------------
Data loading and cleaning functions for the MindMetrix pipeline.

Responsibilities:
    - Load raw CSV files
    - Remove duplicate timestamps
    - Ensure chronological ordering
    - Replace physiologically impossible signal values with NaN

Note: signal-quality flagging (PPG_SQI) and derived signal computation
(gaze velocity) live in features.py, as they are part of feature extraction
rather than data cleaning.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Constants — physiological bounds
# ---------------------------------------------------------------------------

PUPIL_MIN_MM = 0.5   # Below this: tracker loss / blink artefact
PUPIL_MAX_MM = 10.0  # Above this: tracker glitch

HR_MIN_BPM = 30      # Below this: impossible in a conscious resting participant
HR_MAX_BPM = 220     # Above this: physiologically impossible


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(ts_path: str, subjects_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV files from disk.

    Parameters
    ----------
    ts_path : str
        Path to timeseries.csv
    subjects_path : str
        Path to subjects.csv

    Returns
    -------
    ts : pd.DataFrame
        Raw time-series data.
    subjects : pd.DataFrame
        Participant metadata.
    """
    ts = pd.read_csv(ts_path)
    subjects = pd.read_csv(subjects_path)
    print(f"[load] timeseries : {ts.shape[0]:,} rows x {ts.shape[1]} cols")
    print(f"[load] subjects   : {subjects.shape[0]} rows x {subjects.shape[1]} cols")
    return ts, subjects


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with duplicate (SubjectID, DeviceTimestamp) pairs.

    Rationale
    ---------
    The raw data contains exact duplicate rows where the device emitted the
    same sample twice at the same timestamp. Keeping duplicates would inflate
    segment lengths and bias per-phase statistics (mean, std, etc.).
    Only the first occurrence is retained.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["SubjectID", "DeviceTimestamp"])
    n_removed = n_before - len(df)
    if n_removed:
        print(
            f"[clean] Removed {n_removed} duplicate timestamp rows "
            f"({100 * n_removed / n_before:.2f}%)"
        )
    return df.reset_index(drop=True)


def sort_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows by SubjectID → CycleID → DeviceTimestamp.

    Rationale
    ---------
    Chronological ordering within each trial is required for any operation
    that depends on time (diffs, rolling windows, gaze velocity). Sorting
    here once avoids repeating it in every downstream function.
    """
    return df.sort_values(
        ["SubjectID", "CycleID", "DeviceTimestamp"]
    ).reset_index(drop=True)


def clean_pupil_diameter(
    df: pd.DataFrame,
    min_mm: float = PUPIL_MIN_MM,
    max_mm: float = PUPIL_MAX_MM,
) -> pd.DataFrame:
    """
    Replace out-of-range PupilDiameter values with NaN.

    Rationale
    ---------
    The eye-tracker returns negative or near-zero values during blinks or
    when the gaze goes off-screen (tracker loses the pupil). Extremely large
    values indicate sensor glitches. Both are physiologically impossible and
    would corrupt per-phase aggregations if kept.

    Rows are NOT dropped — other signals (HR, motion, gaze) on the same
    timestamp may still be valid.

    Parameters
    ----------
    min_mm : float
        Lower physiological bound (default 0.5 mm).
    max_mm : float
        Upper physiological bound (default 10.0 mm).
    """
    df = df.copy()
    mask = (df["PupilDiameter"] < min_mm) | (df["PupilDiameter"] > max_mm)
    n_invalid = mask.sum()
    df.loc[mask, "PupilDiameter"] = float("nan")
    print(
        f"[clean] PupilDiameter out of [{min_mm}, {max_mm}] mm: "
        f"{n_invalid} rows set to NaN ({100 * n_invalid / len(df):.2f}%)"
    )
    return df


def clean_heart_rate(
    df: pd.DataFrame,
    min_bpm: float = HR_MIN_BPM,
    max_bpm: float = HR_MAX_BPM,
) -> pd.DataFrame:
    """
    Replace out-of-range PulseBPM values with NaN.

    Rationale
    ---------
    Heart rate below 30 bpm is incompatible with a conscious resting state;
    above 220 bpm exceeds the theoretical human maximum. Both indicate PPG
    tracking failure rather than genuine physiology and must be excluded
    before computing any HR-based features.

    Rows are NOT dropped — pupil, gaze, and motion data remain usable.

    Parameters
    ----------
    min_bpm : float
        Lower physiological bound (default 30 bpm).
    max_bpm : float
        Upper physiological bound (default 220 bpm).
    """
    df = df.copy()
    mask = (df["PulseBPM"] < min_bpm) | (df["PulseBPM"] > max_bpm)
    n_invalid = mask.sum()
    df.loc[mask, "PulseBPM"] = float("nan")
    print(
        f"[clean] PulseBPM out of [{min_bpm}, {max_bpm}] bpm: "
        f"{n_invalid} rows set to NaN ({100 * n_invalid / len(df):.2f}%)"
    )
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(ts_path: str, subjects_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Load raw CSVs
    2. Remove duplicate timestamps
    3. Sort chronologically
    4. Replace invalid pupil diameter values with NaN
    5. Replace invalid heart rate values with NaN

    Returns
    -------
    ts : pd.DataFrame
        Cleaned time-series, ready for feature extraction.
    subjects : pd.DataFrame
        Participant metadata (unchanged).
    """
    ts, subjects = load_data(ts_path, subjects_path)
    ts = remove_duplicate_timestamps(ts)
    ts = sort_timeseries(ts)
    ts = clean_pupil_diameter(ts)
    ts = clean_heart_rate(ts)
    return ts, subjects
