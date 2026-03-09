"""
features.py
-----------
Feature engineering for the MindMetrix pipeline.

Pipeline layers
---------------
1. Signal transforms   — add derived columns (GazeVelocity, GazeYaw, GazePitch,
                         hr_valid) to the cleaned time-series before aggregation.
2. Segment aggregators — compute statistics per (SubjectID, CycleID, Phase).
3. Phase pivot         — reshape to one row per (SubjectID, CycleID) with
                         phase-suffixed columns.
4. Relaxation deltas   — compute within-trial changes (relax − baseline).
5. Cycle aggregation   — summarise across the 10 trials per subject
                         (mean, std, linear trend).
6. build_feature_matrix — orchestrates all layers and merges subject metadata.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_KEYS = ["SubjectID", "CycleID", "Phase"]
PHASE_ORDER = ["baseline", "relax", "break"]

PPG_SQI_THRESHOLD = 0.5  # Minimum acceptable signal quality index for HR

# Signals for which relax/break deltas are computed
DELTA_SIGNALS = ["PulseBPM", "PupilDiameter", "MotionMag", "GazeVelocity"]


# ---------------------------------------------------------------------------
# Layer 1 — Signal transforms
# ---------------------------------------------------------------------------

def add_sqi_flag(df: pd.DataFrame, threshold: float = PPG_SQI_THRESHOLD) -> pd.DataFrame:
    """
    Add a boolean column 'hr_valid' to mark rows with acceptable PPG quality.

    Rationale
    ---------
    PPG_SQI (Signal Quality Index) ranges 0–1 and reflects how reliably the
    heart rate was detected. Values below the threshold indicate motion
    artefacts or poor sensor contact. Unlike physiological impossibility
    (handled in preprocessing), low SQI means the device did capture
    something — just unreliably. Flagging rather than nullifying lets us
    keep the column for downstream quality analysis (e.g. hr_valid_ratio
    as a feature in its own right).

    Parameters
    ----------
    threshold : float
        Minimum SQI to consider the HR measurement valid (default 0.5).
    """
    df = df.copy()
    df["hr_valid"] = df["PPG_SQI"] >= threshold
    low_pct = 100 * (~df["hr_valid"]).mean()
    print(
        f"[features] Low-SQI rows (HR excluded from features): "
        f"{(~df['hr_valid']).sum()} ({low_pct:.1f}%)"
    )
    return df


def add_gaze_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'GazeVelocity' column representing instantaneous gaze speed (rad/s).

    Rationale
    ---------
    GazeX/Y/Z are unit-vector components of the 3-D gaze direction. Their
    rate of change over time reflects how quickly the participant's gaze
    moves. High gaze velocity is associated with environmental scanning and
    hypervigilance (common in anxious individuals), whereas stable, slow
    gaze indicates focused attention or a relaxed state.

    Computation
    -----------
    velocity = ||Δgaze|| / Δt   (Euclidean norm of the 3-D gaze difference
                                  divided by the time step in seconds)

    For unit vectors this equals angular velocity in rad/s. Differences are
    computed within each (SubjectID, CycleID, Phase) group so that
    phase-boundary transitions do not produce artifactual spikes. The first
    sample of each group is NaN by definition.
    """
    df = df.copy()
    grp = df.groupby(SEGMENT_KEYS)

    dt = grp["DeviceTimestamp"].diff() / 1e6  # microseconds → seconds
    dx = grp["GazeX"].diff()
    dy = grp["GazeY"].diff()
    dz = grp["GazeZ"].diff()

    df["GazeVelocity"] = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    return df


def add_gaze_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'GazeYaw' and 'GazePitch' columns (radians) from the unit gaze vector.

    Rationale
    ---------
    GazeX/Y/Z encode gaze direction as a 3-D unit vector. Converting to yaw
    and pitch angles yields two independent, interpretable dimensions:

    GazeYaw   = atan2(GazeX, GazeZ)  — horizontal rotation (left ↔ right)
    GazePitch = asin(GazeY)           — vertical rotation (down ↔ up)

    The per-phase standard deviations of these angles (std_yaw, std_pitch)
    quantify how broadly the participant scanned in each plane, without the
    redundancy of keeping X/Y/Z separately. GazeY is clipped to [-1, 1]
    before asin to guard against floating-point values slightly outside the
    unit-sphere that would produce NaN.
    """
    df = df.copy()
    df["GazeYaw"] = np.arctan2(df["GazeX"], df["GazeZ"])
    df["GazePitch"] = np.arcsin(df["GazeY"].clip(-1, 1))
    return df


# ---------------------------------------------------------------------------
# Layer 2 — Segment aggregators  (per SubjectID × CycleID × Phase)
# ---------------------------------------------------------------------------

def _aggregate(
    df: pd.DataFrame,
    signal: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute mean and std of *signal* per segment.

    Parameters
    ----------
    signal : str
        Column name to aggregate.
    mask : pd.Series, optional
        Boolean mask to filter rows before aggregation (e.g. hr_valid).
        The index must align with df.
    """
    data = df if mask is None else df.loc[mask]
    return (
        data.groupby(SEGMENT_KEYS)[signal]
        .agg(mean="mean", std="std")
        .rename(columns=lambda c: f"{signal}_{c}")
        .reset_index()
    )


def extract_hr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heart rate features per segment, restricted to valid-SQI rows.

    Features
    --------
    PulseBPM_mean
        Mean heart rate. Sympathetic nervous system activation (stress,
        anxiety) raises HR; relaxation lowers it via parasympathetic
        dominance. The primary biomarker for relaxation effectiveness.

    PulseBPM_std
        Within-phase HR variability — a crude proxy for heart rate
        variability (HRV). Higher HRV indicates stronger parasympathetic
        tone and is generally associated with better stress regulation.
    """
    if "hr_valid" not in df.columns:
        raise ValueError("Run add_sqi_flag() before extract_hr_features().")
    return _aggregate(df, "PulseBPM", mask=df["hr_valid"])


def extract_pupil_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pupil diameter features per segment.

    Features
    --------
    PupilDiameter_mean
        Mean pupil size. Pupil dilation is controlled by the autonomic
        nervous system: sympathetic activation (anxiety, arousal) causes
        dilation; parasympathetic dominance (relaxation) leads to
        constriction. NaN values (blinks / tracker loss) are automatically
        excluded by pandas aggregations.

    PupilDiameter_std
        Within-phase variability. Elevated std can indicate frequent blink
        artefacts or unstable arousal during the segment.
    """
    return _aggregate(df, "PupilDiameter")


def extract_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Motion magnitude features per segment.

    Features
    --------
    MotionMag_mean
        Average body movement. Anxious participants tend to fidget more,
        producing higher baseline motion. Successful relaxation should
        reduce movement.

    MotionMag_std
        Variability of motion. Captures intermittent bursts of restlessness
        even when the mean is low.
    """
    return _aggregate(df, "MotionMag")


def extract_gaze_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gaze spread and velocity features per segment.

    Features
    --------
    GazeYaw_std
        Standard deviation of horizontal gaze angle (rad). High values
        indicate broad left-right scanning associated with hypervigilance;
        low values reflect stable horizontal fixation typical of a relaxed
        attentional state.

    GazePitch_std
        Standard deviation of vertical gaze angle (rad). Captures up-down
        scanning variability independently of horizontal spread.

    GazeVelocity_mean
        Mean instantaneous gaze speed (rad/s). Rapid gaze shifts suggest
        active environmental monitoring or distraction; slow gaze is
        consistent with focused or relaxed attention.

    GazeVelocity_std
        Variability of gaze speed. Stable, low-variance velocity is
        expected when relaxation is effective.
    """
    if "GazeVelocity" not in df.columns:
        raise ValueError("Run add_gaze_velocity() before extract_gaze_features().")
    if "GazeYaw" not in df.columns or "GazePitch" not in df.columns:
        raise ValueError("Run add_gaze_angles() before extract_gaze_features().")

    grp = df.groupby(SEGMENT_KEYS)
    spread = (
        grp[["GazeYaw", "GazePitch"]]
        .std()
        .rename(columns={"GazeYaw": "GazeYaw_std", "GazePitch": "GazePitch_std"})
        .reset_index()
    )
    velocity = _aggregate(df, "GazeVelocity")
    return spread.merge(velocity, on=SEGMENT_KEYS)


def extract_sqi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PPG signal quality ratio per segment.

    Features
    --------
    hr_valid_ratio
        Fraction of samples with acceptable PPG quality. A low ratio means
        that HR-based features for that segment are derived from few valid
        samples and should be interpreted with caution. Can also serve as
        an indirect motion indicator — high motion degrades PPG quality.

    PPG_SQI_mean
        Mean raw PPG Signal Quality Index (0–1) across the segment.
        Complements hr_valid_ratio by capturing the continuous quality level
        rather than a binary threshold. Useful for correlating signal quality
        with participant characteristics (e.g. WearsGlasses, MotionMag) and
        for flagging segments where HR features are less reliable.
    """
    if "hr_valid" not in df.columns:
        raise ValueError("Run add_sqi_flag() before extract_sqi_features().")
    valid_ratio = (
        df.groupby(SEGMENT_KEYS)["hr_valid"]
        .mean()
        .rename("hr_valid_ratio")
    )
    sqi_mean = (
        df.groupby(SEGMENT_KEYS)["PPG_SQI"]
        .mean()
        .rename("PPG_SQI_mean")
    )
    return pd.concat([valid_ratio, sqi_mean], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Layer 3 — Phase pivot  (per SubjectID × CycleID)
# ---------------------------------------------------------------------------

def pivot_phases(segment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape segment-level features into one row per (SubjectID, CycleID).

    All non-key columns are pivoted using Phase as the column suffix,
    producing names like PulseBPM_mean_baseline, PulseBPM_mean_relax, etc.
    """
    value_cols = [c for c in segment_df.columns if c not in SEGMENT_KEYS]
    pivoted = segment_df.pivot_table(
        index=["SubjectID", "CycleID"],
        columns="Phase",
        values=value_cols,
    )
    pivoted.columns = [f"{feat}_{phase}" for feat, phase in pivoted.columns]
    return pivoted.reset_index()


# ---------------------------------------------------------------------------
# Layer 4 — Relaxation deltas  (per SubjectID × CycleID)
# ---------------------------------------------------------------------------

def compute_relaxation_deltas(
    cycle_df: pd.DataFrame,
    signals: list[str] = DELTA_SIGNALS,
) -> pd.DataFrame:
    """
    Add within-trial change features (relax − baseline, break − baseline).

    Rationale
    ---------
    Absolute signal levels vary considerably between participants due to
    physiology, fitness, and baseline arousal. Using deltas (within-subject
    change) removes this inter-individual variability and isolates the
    physiological response to the relaxation manipulation.

    A negative relax_delta for HR or pupil diameter indicates genuine
    down-regulation during the relaxation phase — the core biomarker of
    relaxation effectiveness.

    The break_delta serves as a recovery proxy: how quickly does the
    participant's physiology return toward baseline after relaxation?

    Features added (per signal in `signals`)
    ----------------------------------------
    <signal>_relax_delta  = mean_relax  − mean_baseline
    <signal>_break_delta  = mean_break  − mean_baseline
    """
    df = cycle_df.copy()
    for sig in signals:
        baseline_col = f"{sig}_mean_baseline"
        relax_col = f"{sig}_mean_relax"
        break_col = f"{sig}_mean_break"
        if baseline_col in df.columns and relax_col in df.columns:
            df[f"{sig}_relax_delta"] = df[relax_col] - df[baseline_col]
        if baseline_col in df.columns and break_col in df.columns:
            df[f"{sig}_break_delta"] = df[break_col] - df[baseline_col]
    return df


# ---------------------------------------------------------------------------
# Layer 5 — Cycle aggregation  (per SubjectID)
# ---------------------------------------------------------------------------

def _linear_trend(group: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Fit a linear slope over trial order for each feature column."""
    x = np.arange(len(group))
    result = {}
    for col in cols:
        y = group[col].values
        valid = ~np.isnan(y)
        result[col] = np.polyfit(x[valid], y[valid], 1)[0] if valid.sum() >= 2 else np.nan
    return pd.Series(result)


def aggregate_across_cycles(
    cycle_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Summarise per-cycle features into one row per subject.

    Aggregations
    ------------
    <feature>_mean
        Average across the 10 trials. Provides a robust estimate of the
        subject's typical physiological response, smoothing out single
        noisy trials.

    <feature>_std
        Across-trial variability. Low std indicates a consistent, reliable
        responder; high std suggests unstable or context-dependent responses.

    <feature>_trend
        Linear slope of the feature over CycleID 1 → 10, estimated by
        least-squares. A negative trend in HR or pupil delta means the
        participant's relaxation response strengthens across repeated trials,
        reflecting physiological adaptation (habituation or learning).
    """
    sorted_df = cycle_df.sort_values(["SubjectID", "CycleID"])

    means = sorted_df.groupby("SubjectID")[feature_cols].mean().add_suffix("_mean")
    stds = sorted_df.groupby("SubjectID")[feature_cols].std().add_suffix("_std")
    trends = (
        sorted_df.groupby("SubjectID")
        .apply(lambda g: _linear_trend(g, feature_cols), include_groups=False)
        .add_suffix("_trend")
    )

    return pd.concat([means, stds, trends], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Layer 6 — Full pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    ts: pd.DataFrame,
    subjects: pd.DataFrame,
) -> pd.DataFrame:
    """
    End-to-end feature extraction pipeline.

    Parameters
    ----------
    ts : pd.DataFrame
        Cleaned time-series output from preprocessing.preprocess().
    subjects : pd.DataFrame
        Participant metadata from subjects.csv.

    Steps
    -----
    1. Apply signal transforms (SQI flag, gaze velocity, gaze angles).
    2. Extract segment-level features per (SubjectID, CycleID, Phase).
    3. Merge all segment feature tables.
    4. Pivot phases to wide format per (SubjectID, CycleID).
    5. Compute relaxation and recovery deltas.
    6. Aggregate across cycles (mean, std, trend per subject).
    7. Merge with participant metadata.

    Returns
    -------
    pd.DataFrame
        One row per subject with all engineered features and metadata columns.
    """
    # --- Step 1: signal transforms ---
    ts = add_sqi_flag(ts)
    ts = add_gaze_velocity(ts)
    ts = add_gaze_angles(ts)

    # --- Step 2: segment-level features ---
    hr = extract_hr_features(ts)
    pupil = extract_pupil_features(ts)
    motion = extract_motion_features(ts)
    gaze = extract_gaze_features(ts)
    sqi = extract_sqi_features(ts)

    # --- Step 3: merge segment tables ---
    segment = (
        hr
        .merge(pupil, on=SEGMENT_KEYS)
        .merge(motion, on=SEGMENT_KEYS)
        .merge(gaze, on=SEGMENT_KEYS)
        .merge(sqi, on=SEGMENT_KEYS)
    )

    # --- Step 4: pivot phases ---
    cycle_wide = pivot_phases(segment)

    # --- Step 5: relaxation deltas ---
    cycle_wide = compute_relaxation_deltas(cycle_wide)

    # --- Step 6: aggregate across cycles ---
    feature_cols = [c for c in cycle_wide.columns if c not in ["SubjectID", "CycleID"]]
    subject_features = aggregate_across_cycles(cycle_wide, feature_cols)

    # --- Step 7: merge metadata ---
    feature_matrix = subject_features.merge(subjects, on="SubjectID")
    print(
        f"[features] Feature matrix: "
        f"{feature_matrix.shape[0]} subjects x {feature_matrix.shape[1]} columns"
    )
    return feature_matrix
