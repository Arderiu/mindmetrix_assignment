"""
analysis.py
-----------
Statistical analysis and unsupervised learning helpers for the MindMetrix pipeline.

Sections
--------
1. Biomarker sensitivity  — paired phase-change tests and effect sizes
2. Metadata correlations  — Spearman correlations and group comparisons
3. PCA                    — standardisation and principal component analysis
4. Clustering             — K-Means optimisation and cluster profiling
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Biomarker sensitivity
# ---------------------------------------------------------------------------

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute paired Cohen's d between two arrays.

    Rationale
    ---------
    Effect size quantifies the *practical* magnitude of a difference,
    independent of sample size. For paired data the appropriate formula is:

        d = mean(a - b) / std(a - b)

    Interpretation: |d| < 0.2 negligible, 0.2 small, 0.5 medium, >= 0.8 large.

    NaN pairs are dropped before computation.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    valid = diff[~np.isnan(diff)]
    if len(valid) < 2 or np.std(valid) == 0:
        return np.nan
    return float(np.mean(valid) / np.std(valid, ddof=1))


def test_phase_change(
    df: pd.DataFrame,
    signal_mean_col_a: str,
    signal_mean_col_b: str,
    label: str = "",
) -> dict:
    """
    Wilcoxon signed-rank test + paired Cohen's d between two per-subject
    feature columns (e.g. mean HR at baseline vs mean HR at relax).

    Rationale
    ---------
    The Wilcoxon signed-rank test is a non-parametric test for dependent
    (paired) samples — suited for before/after or repeated-measures designs
    where each subject provides two related observations (e.g. baseline then
    relax). It is preferred over the paired t-test here because physiological
    features are often not normally distributed across subjects. It tests
    whether the median difference between the two phases is zero without
    assuming Gaussianity.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (one row per subject).
    signal_mean_col_a : str
        Column name for phase A values (e.g. 'PulseBPM_mean_baseline_mean').
    signal_mean_col_b : str
        Column name for phase B values (e.g. 'PulseBPM_mean_relax_mean').
    label : str
        Human-readable label for the result row.

    Returns
    -------
    dict with keys: label, n, median_a, median_b, median_delta, statistic,
                    p_value, cohens_d
    """
    a = df[signal_mean_col_a].dropna()
    b = df[signal_mean_col_b].dropna()
    common_idx = a.index.intersection(b.index)
    a, b = a.loc[common_idx].values, b.loc[common_idx].values

    result = stats.wilcoxon(a, b, nan_policy="omit")
    d = cohens_d_paired(a, b)

    return {
        "label": label or f"{signal_mean_col_a} vs {signal_mean_col_b}",
        "n": len(a),
        "median_a": float(np.nanmedian(a)),
        "median_b": float(np.nanmedian(b)),
        "median_delta": float(np.nanmedian(b - a)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohens_d": d,
    }


def test_trend(
    df: pd.DataFrame,
    col: str,
    label: str = "",
) -> dict:
    """
    One-sample Wilcoxon signed-rank test: is the cross-trial slope ≠ 0?

    Rationale
    ---------
    The trend columns (e.g. PulseBPM_relax_delta_trend) contain the linear
    slope of the relaxation delta across the 10 trials for each subject.
    A significant negative slope in HR or pupil delta would indicate that
    participants' relaxation response strengthens with practice. Testing
    against zero is a one-sample signed-rank test: no reference distribution
    is assumed, and the null is that the median slope is zero.

    Cohen's d here is the one-sample version: mean(slope) / std(slope),
    capturing how consistently non-zero the trend is across subjects.

    Parameters
    ----------
    col : str
        Trend column to test (e.g. 'PulseBPM_relax_delta_trend').
    label : str
        Human-readable label for the result row.
    """
    x = df[col].dropna().values
    result = stats.wilcoxon(x, nan_policy="omit")
    d = (float(np.mean(x) / np.std(x, ddof=1))
         if len(x) >= 2 and np.std(x, ddof=1) > 0 else np.nan)
    return {
        "label": label or col,
        "n": len(x),
        "median_trend": float(np.median(x)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohens_d": d,
        "significant": result.pvalue < 0.05,
    }


def biomarker_sensitivity_report(
    df: pd.DataFrame,
    tests: list[dict],
) -> pd.DataFrame:
    """
    Run sensitivity tests for multiple signals and return a unified summary table.

    Each entry in `tests` can be either a phase-change test or a trend test,
    detected from the keys present:

    Phase-change test (keys: label, col_a, col_b)
        Paired Wilcoxon signed-rank between two phases (e.g. baseline → relax).
        Populates: median_a, median_b, median_delta.

    Trend test (keys: label, col)
        One-sample Wilcoxon signed-rank against zero — tests whether the
        linear slope of the relaxation response across the 10 trials is
        significantly different from zero (learning / habituation effect).
        Populates: median_trend.

    Example
    -------
    tests = [
        {"label": "HR baseline→relax",
         "col_a": "PulseBPM_mean_baseline_mean",
         "col_b": "PulseBPM_mean_relax_mean"},
        {"label": "HR trend across trials",
         "col": "PulseBPM_relax_delta_trend"},
    ]
    report = biomarker_sensitivity_report(feature_matrix, tests)

    Returns
    -------
    pd.DataFrame sorted by absolute Cohen's d (strongest effects first).
    """
    rows = []
    for t in tests:
        if "col_a" in t and "col_b" in t:
            row = test_phase_change(df, t["col_a"], t["col_b"], label=t["label"])
        else:
            row = test_trend(df, t["col"], label=t["label"])
        rows.append(row)

    report = pd.DataFrame(rows)
    report["significant"] = report["p_value"] < 0.05
    report["abs_cohens_d"] = report["cohens_d"].abs()
    return report.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Metadata correlations
# ---------------------------------------------------------------------------

def spearman_with_pvalues(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman rank correlation between physiological features and
    continuous metadata targets (e.g. STAI_T, STAI_S).

    Rationale
    ---------
    Spearman is preferred over Pearson here because:
    - Physiological features are not guaranteed to be normally distributed.
    - Spearman captures monotonic relationships without assuming linearity.
    - It is more robust to outliers in both features and STAI scores.

    Parameters
    ----------
    feature_cols : list[str]
        Physiological feature columns to correlate.
    target_cols : list[str]
        Metadata columns to correlate against (e.g. ['STAI_T', 'STAI_S']).

    Returns
    -------
    corr_df : pd.DataFrame
        Spearman r values (features × targets).
    pval_df : pd.DataFrame
        Corresponding p-values.
    """
    corr_data, pval_data = {}, {}
    for target in target_cols:
        rs, ps = [], []
        for feat in feature_cols:
            valid = df[[feat, target]].dropna()
            if len(valid) < 5:
                rs.append(np.nan)
                ps.append(np.nan)
            else:
                r, p = stats.spearmanr(valid[feat], valid[target])
                rs.append(float(r))
                ps.append(float(p))
        corr_data[target] = rs
        pval_data[target] = ps

    corr_df = pd.DataFrame(corr_data, index=feature_cols)
    pval_df = pd.DataFrame(pval_data, index=feature_cols)
    return corr_df, pval_df


def mann_whitney_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
) -> pd.DataFrame:
    """
    Mann-Whitney U test comparing feature distributions between two groups.

    Rationale
    ---------
    Mann-Whitney U is the non-parametric equivalent of the independent
    samples t-test. It tests whether one group tends to have higher values
    than the other without assuming normality, making it suitable for
    physiological features with skewed distributions.

    Use this for binary metadata columns (Gender, Handedness, WearsGlasses).

    Parameters
    ----------
    group_col : str
        Binary categorical column defining the two groups.

    Returns
    -------
    pd.DataFrame with columns: feature, group_a, group_b, median_a, median_b,
                                statistic, p_value, significant
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"mann_whitney_report expects exactly 2 groups in '{group_col}', "
            f"found: {groups}"
        )
    g_a, g_b = sorted(groups)
    rows = []
    for feat in feature_cols:
        a = df.loc[df[group_col] == g_a, feat].dropna().values
        b = df.loc[df[group_col] == g_b, feat].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "feature": feat,
            "group_a": g_a,
            "group_b": g_b,
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "significant": result.pvalue < 0.05,
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. PCA
# ---------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, StandardScaler]:
    """
    Z-score standardise selected feature columns.

    Rationale
    ---------
    PCA and K-Means are both sensitive to feature scale. Standardisation
    ensures that signals measured in different units (bpm, mm, rad/s) and
    at different magnitudes contribute equally to the analysis rather than
    having high-variance features dominate.

    NaN values are imputed with each feature's column mean before scaling
    so that subjects with missing values are retained in the analysis.

    Returns
    -------
    X_scaled : np.ndarray  shape (n_subjects, n_features)
    scaler   : fitted StandardScaler (for inverse-transforming later)
    """
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_pca(
    X_scaled: np.ndarray,
    feature_cols: list[str],
    n_components: int | None = None,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """
    Fit PCA on standardised features.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardised feature matrix from scale_features().
    feature_cols : list[str]
        Original feature names (used to label loadings).
    n_components : int, optional
        Number of components to retain. If None, all are kept.

    Returns
    -------
    pca : fitted PCA object
    scores_df : pd.DataFrame
        PCA scores (projections) for each subject — shape (n_subjects, n_components).
    variance_df : pd.DataFrame
        Explained variance ratio and cumulative variance per component.
    loadings_df : pd.DataFrame
        Component loadings (n_components × n_features) — shows which original
        features drive each principal component.
    """
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    comp_labels = [f"PC{i+1}" for i in range(pca.n_components_)]

    scores_df = pd.DataFrame(scores, columns=comp_labels)

    variance_df = pd.DataFrame({
        "component": comp_labels,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    loadings_df = pd.DataFrame(
        pca.components_,
        index=comp_labels,
        columns=feature_cols,
    )

    return pca, scores_df, variance_df, loadings_df


# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------

def elbow_and_silhouette(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute K-Means inertia and silhouette score for a range of K values.

    Rationale
    ---------
    Two complementary criteria are used to select the optimal number of clusters:

    Inertia (elbow method)
        Sum of squared distances from each point to its cluster centre.
        Plotted against K, the curve typically shows an "elbow" where adding
        more clusters yields diminishing returns — the optimal K sits at the
        bend.

    Silhouette score
        Measures how similar each point is to its own cluster compared to
        other clusters (range −1 to 1). Higher is better; the K with the
        highest silhouette is the most compact and well-separated clustering.

    Using both together avoids over-relying on a single heuristic.

    Returns
    -------
    pd.DataFrame with columns: k, inertia, silhouette_score
    """
    from sklearn.metrics import silhouette_score

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})

    return pd.DataFrame(rows)


def run_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-Means with the chosen number of clusters.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (output of scale_features or PCA scores).
    k : int
        Number of clusters selected from elbow_and_silhouette analysis.

    Returns
    -------
    km : fitted KMeans object
    labels : np.ndarray of cluster assignments, shape (n_subjects,)
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    print(f"[clustering] K-Means (k={k}) — cluster sizes: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")
    return km, labels


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    cols: list[str],
) -> pd.DataFrame:
    """
    Summarise cluster membership by computing the mean of selected columns
    per cluster.

    Rationale
    ---------
    After clustering on physiological features, profiling with metadata
    (STAI_T, STAI_S, Gender, etc.) and key biomarkers (relax_delta)
    reveals whether clusters correspond to meaningful participant groups
    (e.g. strong vs weak relaxation responders, high vs low anxiety).

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (one row per subject, same order as labels).
    labels : np.ndarray
        Cluster assignments from run_kmeans().
    cols : list[str]
        Columns to include in the profile (mix of features and metadata).

    Returns
    -------
    pd.DataFrame with one row per cluster and one column per selected variable.
    """
    profiled = df[cols].copy()
    profiled["cluster"] = labels
    return (
        profiled.groupby("cluster")[cols]
        .agg(["mean", "std"])
        .round(3)
    )
