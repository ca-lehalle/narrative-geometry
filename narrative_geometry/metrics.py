"""
Statistical metrics, weekly resampling, prevalence estimation, covariation matrices,
bootstrap specificity confidence intervals, and Welch t-test + BH-FDR rupture detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DIFF_COL_MAP = {"p_delta_V": "diff_V", "p_delta_N": "diff_N", "p_diff": "diff_diff"}


def resample_constant_weekly(
    df: pd.DataFrame,
    target_per_week: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Resample a DataFrame to have a constant number of articles per week,
    while keeping the relative proportions of categories.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Create week identifier
    df["week"] = df["date"].dt.to_period("W")

    # 2. Target size per week
    weekly_counts = df.groupby("week").size()
    if target_per_week is None:
        target_per_week = int(weekly_counts.median())

    print(f"{'Semaines totales':<35}: {len(weekly_counts)}")
    print(f"{'Articles/semaine (min / méd / max)':<35}: {weekly_counts.min()} / {int(weekly_counts.median())} / {weekly_counts.max()}")
    print(f"{'Nombre cible par semaine':<35}: {target_per_week}")

    rng = np.random.default_rng(random_state)
    sampled_weeks = []

    # 3. For each week, sample proportionally per category
    for week, group in df.groupby("week"):
        n_week = len(group)
        cat_counts = group["category"].value_counts()
        props = cat_counts / n_week

        raw_alloc = props * target_per_week
        alloc = np.floor(raw_alloc).astype(int)

        remainder = target_per_week - alloc.sum()
        if remainder > 0:
            fractional_parts = raw_alloc - alloc
            top_cats = fractional_parts.nlargest(int(remainder)).index
            alloc[top_cats] += 1

        week_samples = []
        for cat, n_target in alloc.items():
            if n_target <= 0:
                continue
            cat_group = group[group["category"] == cat]
            replace = n_target > len(cat_group)
            idx = rng.choice(cat_group.index, size=n_target, replace=replace)
            week_samples.append(df.loc[idx])

        if week_samples:
            sampled_weeks.append(pd.concat(week_samples, axis=0))

    result = pd.concat(sampled_weeks, axis=0).drop(columns=["week"]).reset_index(drop=True)
    print(f"{'Total articles après ré-échantillonnage':<35}: {len(result)}")
    return result


def pairwise_mean_product_matrix(Q: pd.DataFrame, labels) -> pd.DataFrame:
    """Compute (1/T) * Q^T @ Q pairwise second-moment co-variation matrix."""
    A = Q.to_numpy(dtype=float)
    M = (A.T @ A) / A.shape[0]
    return pd.DataFrame(M, index=labels, columns=labels)


def upper_triangle_values(df: pd.DataFrame) -> np.ndarray:
    """Extract strictly upper-triangular off-diagonal values from a square DataFrame."""
    mask = np.triu(np.ones(df.shape, dtype=bool), k=1)
    return df.where(mask).stack().values


def df_to_latex_booktabs(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.3f") -> str:
    """Format a DataFrame as a LaTeX booktabs table."""
    n_cols = len(df.columns)
    col_format = "l" + "c" * n_cols
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        " & ".join([""] + list(df.columns)) + r" \\",
        r"\midrule",
    ]
    for idx, row in df.iterrows():
        values = " & ".join(float_fmt % v for v in row.values)
        lines.append(f"{idx} & {values} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def delta(df: pd.DataFrame, chapter: str, narrative_name: str) -> tuple[float, float]:
    """Compute in-chapter minus out-of-chapter mean similarity for N and V."""
    in_mask = df["category"].str.lower() == chapter
    out_mask = df["category"].str.lower() != chapter
    avg_N_in = df[in_mask][f"{chapter}:{narrative_name}"].mean()
    avg_V_in = df[in_mask][f"{chapter}:{narrative_name}_orth"].mean()
    avg_N_out = df[out_mask][f"{chapter}:{narrative_name}"].mean()
    avg_V_out = df[out_mask][f"{chapter}:{narrative_name}_orth"].mean()
    return avg_N_in - avg_N_out, avg_V_in - avg_V_out


def bootstrap_delta_ci(
    df: pd.DataFrame,
    chapter: str,
    narrative_name: str,
    n_bootstrap: int = 100,
    frac: float = 0.2,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap confidence intervals for Delta(N), Delta(V), and Delta_gain."""
    rng = np.random.default_rng(42)

    chapter_df = df[df["category"].str.lower() == chapter].copy()
    out_df = df[df["category"].str.lower() != chapter].copy()

    n_in = len(chapter_df)
    n_out = len(out_df)
    m_in = max(1, int(n_in * frac))
    m_out = max(1, int(n_out * frac))

    Delta_N = np.empty(n_bootstrap, dtype=float)
    Delta_V = np.empty(n_bootstrap, dtype=float)
    Gain = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        in_idx = rng.choice(np.arange(n_in), size=m_in, replace=True)
        out_idx = rng.choice(np.arange(n_out), size=m_out, replace=True)

        sample_in = chapter_df.iloc[in_idx]
        sample_out = out_df.iloc[out_idx]

        avg_N_in = sample_in[f"{chapter}:{narrative_name}"].mean()
        avg_V_in = sample_in[f"{chapter}:{narrative_name}_orth"].mean()
        avg_N_out = sample_out[f"{chapter}:{narrative_name}"].mean()
        avg_V_out = sample_out[f"{chapter}:{narrative_name}_orth"].mean()

        d_N = avg_N_in - avg_N_out
        d_V = avg_V_in - avg_V_out

        Delta_N[b] = d_N
        Delta_V[b] = d_V
        Gain[b] = d_V - d_N

    return {
        "mean_delta_N": Delta_N.mean(),
        "mean_delta_V": Delta_V.mean(),
        "mean_gain": Gain.mean(),
        "ci_delta_N_low": np.quantile(Delta_N, alpha / 2),
        "ci_delta_N_high": np.quantile(Delta_N, 1 - alpha / 2),
        "ci_delta_V_low": np.quantile(Delta_V, alpha / 2),
        "ci_delta_V_high": np.quantile(Delta_V, 1 - alpha / 2),
        "ci_gain_low": np.quantile(Gain, alpha / 2),
        "ci_gain_high": np.quantile(Gain, 1 - alpha / 2),
    }


def fmt_ci(mean: float, lo: float, hi: float, sig: bool | None = None) -> str:
    """Format point estimate and confidence interval for display."""
    star = "$^{*}$" if sig else ""
    return f"{mean:.4f}{star} [{lo:.4f}, {hi:.4f}]"


def run_moving_tests_fdr(
    embeddings: list,
    dates,
    N: np.ndarray,
    V: np.ndarray,
    t_begin: pd.Timestamp,
    t_end: pd.Timestamp,
    window_size: int = 1,
    bandwidth_days: int = 90,
) -> pd.DataFrame:
    """
    For each day t in [t_begin, t_end], runs three two-sided Welch t-tests
    comparing `bandwidth_days` after t to `bandwidth_days` before t, with
    Benjamini-Hochberg FDR correction.
    """
    dates_s = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    scores_V = np.array([x @ V for x in embeddings], dtype=float)
    scores_N = np.array([x @ N for x in embeddings], dtype=float)
    scores = {"p_delta_V": scores_V, "p_delta_N": scores_N, "p_diff": scores_V - scores_N}

    window, bandwidth = pd.Timedelta(days=window_size), pd.Timedelta(days=bandwidth_days)
    rows = []
    for t in pd.date_range(start=t_begin, end=t_end, freq="D"):
        before = ((dates_s >= t - bandwidth - window) & (dates_s < t - window)).to_numpy()
        after = ((dates_s > t + window) & (dates_s <= t + bandwidth + window)).to_numpy()
        if min(before.sum(), after.sum()) < 2:
            continue
        p_vals, diffs = {}, {}
        for p_col, s in scores.items():
            p_vals[p_col] = stats.ttest_ind(s[after], s[before], alternative="two-sided", equal_var=False)[1]
            diffs[DIFF_COL_MAP[p_col]] = s[after].mean() - s[before].mean()
        rows.append({"date": t, **p_vals, **diffs})

    df = pd.DataFrame(rows)
    for col in scores:
        df[f"{col}_fdr"] = multipletests(df[col], method="fdr_bh")[1]
    return df


def detect_ruptures(
    results: pd.DataFrame,
    test_col: str = "p_delta_V",
    alpha: float = 0.05,
    trim_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim edge days and group consecutive FDR-significant days into detected ruptures."""
    fdr_col, diff_col = f"{test_col}_fdr", DIFF_COL_MAP[test_col]
    df = results.copy().sort_values("date").reset_index(drop=True)

    t_min, t_max = df["date"].min(), df["date"].max()
    df = df[
        (df["date"] >= t_min + pd.Timedelta(days=trim_days))
        & (df["date"] <= t_max - pd.Timedelta(days=trim_days))
    ].reset_index(drop=True)

    df["significant"] = df[fdr_col] < alpha
    df["group"] = (df["significant"] != df["significant"].shift()).cumsum()

    events = []
    for _, g in df[df["significant"]].groupby("group"):
        best = g.loc[g[test_col].idxmin()]
        events.append({
            "rupture_date": best["date"],
            "run_start": g["date"].min(),
            "run_end": g["date"].max(),
            "p_raw": best[test_col],
            "p_fdr": best[fdr_col],
            "direction": "+" if best[diff_col] > 0 else "-",
            "magnitude": best[diff_col],
            "n_consecutive_days": len(g),
        })
    cols = [
        "rupture_date",
        "run_start",
        "run_end",
        "p_raw",
        "p_fdr",
        "direction",
        "magnitude",
        "n_consecutive_days",
    ]
    return df, pd.DataFrame(events, columns=cols).sort_values("rupture_date").reset_index(drop=True)


def weekly_ema(scores, dates, start, end, span_weeks: int = 4) -> pd.Series:
    """Compute weekly mean of article scores smoothed with an EMA."""
    s = pd.Series(scores, index=pd.to_datetime(dates)).groupby(pd.Grouper(freq="W-MON")).mean()
    return s[(s.index >= start) & (s.index <= end)].ewm(span=span_weeks, adjust=False).mean()


def pairwise_jaccard(sig_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity index between boolean significance series."""
    labels = sig_matrix.columns
    n = len(labels)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = sig_matrix.iloc[:, i].values, sig_matrix.iloc[:, j].values
            union = (a | b).sum()
            mat[i, j] = (a & b).sum() / union if union > 0 else np.nan
    return pd.DataFrame(mat, index=labels, columns=labels)
