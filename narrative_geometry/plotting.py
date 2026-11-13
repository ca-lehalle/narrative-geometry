"""
Plotting utilities for cosine similarity matrices, chapter-grouped heatmaps,
EMA prevalence time series, and detected rupture visualizations.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from IPython.display import display

from .metrics import detect_ruptures, run_moving_tests_fdr, weekly_ema

SPECTRAL_CMAP = LinearSegmentedColormap.from_list(
    "spectral_contrast",
    [
        "#2d004b",
        "#4a0080",
        "#0d47a1",
        "#1976d2",
        "#90caf9",
        "#ffffff",
        "#fff176",
        "#ffb300",
        "#e64a19",
        "#b71c1c",
        "#4a0000",
    ],
    N=512,
)


def short_label(title: str, maxw: int = 16) -> str:
    """Wrap a narrative title into at most two short lines for heatmap tick labels."""
    words = title.replace("_", " ").split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) > maxw:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines[:2])


def draw_chapter_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    chapter_names: list[str],
    chapter_pal: dict[str, str],
    chapter_abbrev: dict[str, str],
    chapter_size: int = 3,
    vmin: float = -1,
    vmax: float = 1,
    annot: bool = True,
    cmap=SPECTRAL_CMAP,
):
    """Draw a cosine similarity heatmap with thematic chapter shading and borders."""
    nr, nc = matrix.shape
    n_total = len(chapter_names) * chapter_size

    # Background chapter shading
    for i, ch in enumerate(chapter_names):
        c = chapter_pal[ch]
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, i * chapter_size - 0.5),
                nc,
                chapter_size,
                color=c,
                alpha=0.08,
                linewidth=0,
                zorder=0,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (i * chapter_size - 0.5, -0.5),
                chapter_size,
                nr,
                color=c,
                alpha=0.08,
                linewidth=0,
                zorder=0,
            )
        )

    display_matrix = matrix * 2 - 1 if vmin == 0 else matrix
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)

    im = ax.imshow(
        display_matrix,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="none",
        zorder=1,
    )

    # Chapter block borders
    for i, ch in enumerate(chapter_names):
        c = chapter_pal[ch]
        s = i * chapter_size
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, s - 0.5),
                nc,
                chapter_size,
                edgecolor=c,
                facecolor="none",
                linewidth=1.5,
                zorder=3,
                alpha=1.0,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (s - 0.5, -0.5),
                chapter_size,
                nr,
                edgecolor=c,
                facecolor="none",
                linewidth=1.5,
                zorder=3,
                alpha=1.0,
            )
        )

    # White separators
    for k in range(1, max(1, n_total // chapter_size)):
        pos = k * chapter_size - 0.5
        ax.axvline(pos, color="white", lw=2.0, zorder=4)
        ax.axhline(pos, color="white", lw=2.0, zorder=4)

    # Cell annotations
    if annot:
        for i in range(nr):
            for j in range(nc):
                v = matrix[i, j]
                if v < -0.20:
                    fg = "white"
                elif v < 0.10:
                    fg = "#222222"
                elif v < 0.38:
                    fg = "#111111"
                else:
                    fg = "white"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=fg,
                    fontfamily="monospace",
                    fontweight="bold",
                    zorder=5,
                )

    ax.set_xticks(range(nc))
    ax.set_xticklabels(
        col_labels, fontsize=5.8, rotation=40, ha="right", va="top", linespacing=1.3
    )
    ax.set_yticks(range(nr))
    ax.set_yticklabels(row_labels, fontsize=5.8, va="center", linespacing=1.3)
    ax.tick_params(length=0, pad=4)

    for i, ch in enumerate(chapter_names):
        mid = i * chapter_size + (chapter_size - 1) / 2
        col = chapter_pal[ch]
        name = chapter_abbrev[ch]
        ax.annotate(
            name,
            xy=(-0.5, mid),
            xycoords="data",
            xytext=(-60, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=col,
            annotation_clip=False,
            rotation=90,
        )

    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=10, loc="center")
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    return im


def add_colorbar(fig, im, ax, vmin: float, vmax: float):
    """Attach a formatted colorbar to a heatmap axis."""
    cbar_ax = ax.inset_axes([1.04, 0.0, 0.025, 1.0])
    ticks = np.linspace(vmin, vmax, 5)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=np.linspace(-1, 1, 5))
    cb.ax.set_yticklabels([f"{v:.1f}" for v in ticks], fontsize=6)
    cb.set_label("Cosine similarity", fontsize=6.5, labelpad=6)
    cb.outline.set_linewidth(0.4)


def plot_narrative_timeseries(
    cols: list[str],
    chapters: list[str],
    ts_df: pd.DataFrame,
    span_weeks: int = 4,
    strip_suffix: str = "",
    title: str = "",
    start: str = "2014-01-01",
    end: str = "2018-12-31",
    events: dict[str, str] | None = None,
    figsize: tuple[int, int] = (16, 8),
):
    """Plot weekly EMA time series of narrative prevalence across chapters."""
    ts_df_period = ts_df[(ts_df["date_dt"] >= start) & (ts_df["date_dt"] <= end)].copy()
    fig, ax = plt.subplots(figsize=figsize)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    palette = plt.cm.tab20(np.linspace(0, 1, max(10, len(cols))))

    n = 0
    for chapter in chapters:
        for col in [c for c in cols if c.startswith(f"{chapter}:")]:
            weekly_mean = ts_df_period.groupby(pd.Grouper(key="date_dt", freq="W-MON"))[col].mean()
            smoothed = weekly_mean.ewm(span=span_weeks, adjust=False).mean()

            narrative_name = col.split(":")[1].replace(strip_suffix, "").replace("_", " ")
            ax.plot(
                smoothed.index,
                smoothed.values,
                label=f"{chapter.capitalize()} - {narrative_name}",
                color=palette[n % len(palette)],
                linewidth=2.2,
            )
            n += 1

    if events is not None:
        event_texts = []
        for _, (date_str, label) in enumerate(sorted(events.items())):
            x = pd.Timestamp(date_str)
            ax.axvline(x=x, color="red", linestyle="--", linewidth=1.4, alpha=0.9)

            text_obj = ax.text(
                x,
                ax.get_ylim()[1] * 0.96,
                label,
                rotation=90,
                fontsize=8,
                color="red",
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="red",
                    alpha=0.8,
                    linewidth=0.8,
                ),
            )
            event_texts.append(text_obj)

        adjust_text(
            event_texts,
            ax=ax,
            only_move={"points": "y", "text": "y"},
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Weekly Mean $\\hat{p}$ (EMA)", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

    plt.tight_layout()
    plt.show()


def _draw_narrative_on_ax(
    ax,
    embeddings,
    dates,
    label: str,
    get_vectors_fn: Callable[[str], tuple[np.ndarray, np.ndarray]],
    t_begin: pd.Timestamp,
    t_end: pd.Timestamp,
    color_N: str = "steelblue",
    color_V: str = "firebrick",
    bandwidth_days: int = 90,
    alpha: float = 0.05,
    trim_days: int = 90,
    span_weeks: int = 4,
    start=None,
    end=None,
    bar_linewidth: int = 8,
    bar_alpha: float = 0.45,
    y_top: float = 1.10,
    y_bottom: float = -0.14,
    events=None,
):
    start = t_begin if start is None else start
    end = t_end if end is None else end

    N, V = get_vectors_fn(label)

    results = run_moving_tests_fdr(
        embeddings, dates, N, V, t_begin, t_end, bandwidth_days=bandwidth_days
    )
    _, ruptures_N = detect_ruptures(results, test_col="p_delta_N", alpha=alpha, trim_days=trim_days)
    _, ruptures_V = detect_ruptures(results, test_col="p_delta_V", alpha=alpha, trim_days=trim_days)

    s_N = weekly_ema(np.array([x @ N for x in embeddings], dtype=float), dates, start, end, span_weeks)
    s_V = weekly_ema(np.array([x @ V for x in embeddings], dtype=float), dates, start, end, span_weeks)
    s_V_rescaled = (s_V - s_V.min()) / (s_V.max() - s_V.min()) * (s_N.max() - s_N.min()) + s_N.min()

    ax.plot(s_N.index, s_N.values, color=color_N, linewidth=1.5, linestyle="-", label="N")
    ax.plot(
        s_V_rescaled.index,
        s_V_rescaled.values,
        color=color_V,
        linewidth=1.5,
        linestyle="--",
        label="V (rééchelonné)",
    )

    trans = ax.get_xaxis_transform()
    for ruptures, color, y in [(ruptures_N, color_N, y_top), (ruptures_V, color_V, y_bottom)]:
        for _, row in ruptures.iterrows():
            ax.plot(
                [row["run_start"], row["run_end"]],
                [y, y],
                color=color,
                linewidth=bar_linewidth,
                alpha=bar_alpha,
                solid_capstyle="butt",
                transform=trans,
                clip_on=False,
            )

    if events is not None:
        for date_str, _, _ in events:
            date = pd.to_datetime(date_str)
            if date < start or date > end:
                continue
            ax.axvline(date, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    return ruptures_N, ruptures_V


def plot_narratives_ruptures(
    embeddings,
    dates,
    labels: list[str] | str,
    t_begin: pd.Timestamp,
    t_end: pd.Timestamp,
    get_narrative_vectors: Callable[[str], tuple[np.ndarray, np.ndarray]] | None = None,
    color_N: str = "steelblue",
    color_V: str = "firebrick",
    bandwidth_days: int = 90,
    alpha: float = 0.05,
    trim_days: int = 90,
    span_weeks: int = 4,
    start=None,
    end=None,
    events=None,
):
    """Plot small multiples of N vs V (rescaled) with FDR-corrected detected ruptures."""
    if get_narrative_vectors is None:
        raise ValueError("get_narrative_vectors callable is required.")
    if isinstance(labels, str):
        labels = [labels]
    n = len(labels)

    fig, axes = plt.subplots(
        n, 1, figsize=(16, 5.5 * n), sharex=False, gridspec_kw={"hspace": 0.9}
    )
    if n == 1:
        axes = [axes]

    all_ruptures = {}
    for ax, label in zip(axes, labels):
        all_ruptures[label] = _draw_narrative_on_ax(
            ax,
            embeddings,
            dates,
            label,
            get_narrative_vectors,
            t_begin,
            t_end,
            color_N=color_N,
            color_V=color_V,
            bandwidth_days=bandwidth_days,
            alpha=alpha,
            trim_days=trim_days,
            span_weeks=span_weeks,
            start=start,
            end=end,
            events=events,
        )
        ax.set_title(label, fontsize=12, fontweight="bold", pad=45)
        ax.set_ylabel("Score (N ; V rescaled)", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=12)

    handles = [
        Line2D([0], [0], color=color_N, lw=2, linestyle="-", label="N (raw)"),
        Line2D([0], [0], color=color_V, lw=2, linestyle="--", label="V (rescaled)"),
        Line2D([0], [0], color=color_N, lw=6, alpha=0.45, label="Rupture on N"),
        Line2D([0], [0], color=color_V, lw=6, alpha=0.45, label="Rupture on V"),
    ]

    if events is not None:
        for date_str, _, _ in events:
            date = pd.to_datetime(date_str)
            axes[-1].axvline(date, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)

    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "Detected ruptures (FDR-corrected) — N vs V (rescaled)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    return all_ruptures


def plot_narrative_vs_events(
    embeddings,
    dates,
    label: str,
    events: list[tuple[str, str, str]],
    t_begin: pd.Timestamp,
    t_end: pd.Timestamp,
    get_narrative_vectors: Callable[[str], tuple[np.ndarray, np.ndarray]] | None = None,
    kinds: tuple[str, ...] = ("V", "N"),
    span_weeks: int = 4,
    bandwidth_days: int = 90,
    alpha: float = 0.05,
    trim_days: int = 90,
    min_n_days: int = 1,
):
    """Plot EMA series of V and/or N with detected rupture bands and exogenous events."""
    if get_narrative_vectors is None:
        raise ValueError("get_narrative_vectors callable is required.")
    N, V = get_narrative_vectors(label)
    results = run_moving_tests_fdr(
        embeddings, dates, N, V, t_begin, t_end, bandwidth_days=bandwidth_days
    )
    spec = {
        "V": (V, "p_delta_V", "firebrick", "Narratif purifié (V)", "pure"),
        "N": (N, "p_delta_N", "steelblue", "Narratif brut (N)", "raw"),
    }
    colors = plt.cm.tab10.colors
    name = label.split(":", 1)[-1]

    for kind in kinds:
        vec, test_col, line_color, line_label, tag = spec[kind]
        _, ruptures = detect_ruptures(results, test_col=test_col, alpha=alpha, trim_days=trim_days)
        print(f"{len(ruptures)} ruptures detected on narrative {tag} ({kind}).")

        smoothed = weekly_ema(
            np.array([x @ vec for x in embeddings], dtype=float),
            dates,
            t_begin,
            t_end,
            span_weeks,
        )

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(smoothed.index, smoothed.values, color=line_color, linewidth=1.5, label=line_label)

        to_plot = ruptures[ruptures["n_consecutive_days"] >= min_n_days].reset_index(drop=True)
        for i, row in to_plot.iterrows():
            color = colors[i % len(colors)]
            ax.axvspan(row["run_start"], row["run_end"], color=color, alpha=0.15)
            ax.axvline(row["rupture_date"], color=color, linestyle=":", linewidth=1.0, alpha=0.8)
            mid = row["run_start"] + (row["run_end"] - row["run_start"]) / 2
            ax.annotate(
                f"{row['n_consecutive_days']}j — p={row['p_fdr']:.1e}",
                xy=(mid, ax.get_ylim()[1]),
                xytext=(0, -10),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="center",
                fontsize=7,
                color=color,
            )

        for date_str, ev_label, _ in events:
            date = pd.to_datetime(date_str)
            if date < smoothed.index.min() or date > smoothed.index.max():
                continue
            ax.axvline(date, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)
            ax.text(
                date,
                ax.get_ylim()[0],
                ev_label,
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=7,
                color="dimgray",
            )

        ax.set_title(
            f"Narrative {name} ({tag}) — detected ruptures vs. exogenous events",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Weekly Mean $\\hat{p}$ (EMA)", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    display(pd.DataFrame(events, columns=["date", "label", "description"]))
