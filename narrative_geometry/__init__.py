"""
narrative_geometry: Algebraic orthogonalization on the Stiefel manifold for
monitoring the relative dominance of competing narratives.
"""

from .embeddings import average_cluster_prototypes, get_embedding, load_model
from .io_utils import (
    find_freshest_dated_file,
    get_repo_root,
    get_resources_dir,
    resolve_storage_dir,
    save_and_download_figure,
)
from .manifold import (
    decompose_narratives,
    gram_schmidt,
    normalize_rows,
    procrustes_stiefel,
    stiefel_optimize,
)
from .metrics import (
    DIFF_COL_MAP,
    bootstrap_delta_ci,
    delta,
    detect_ruptures,
    df_to_latex_booktabs,
    fmt_ci,
    pairwise_jaccard,
    pairwise_mean_product_matrix,
    resample_constant_weekly,
    run_moving_tests_fdr,
    upper_triangle_values,
    weekly_ema,
)
from .plotting import (
    SPECTRAL_CMAP,
    add_colorbar,
    draw_chapter_heatmap,
    plot_narrative_timeseries,
    plot_narrative_vs_events,
    plot_narratives_ruptures,
    short_label,
)

__all__ = [
    "DIFF_COL_MAP",
    "SPECTRAL_CMAP",
    "add_colorbar",
    "average_cluster_prototypes",
    "bootstrap_delta_ci",
    "decompose_narratives",
    "delta",
    "detect_ruptures",
    "df_to_latex_booktabs",
    "draw_chapter_heatmap",
    "find_freshest_dated_file",
    "fmt_ci",
    "get_embedding",
    "get_repo_root",
    "get_resources_dir",
    "gram_schmidt",
    "load_model",
    "normalize_rows",
    "pairwise_jaccard",
    "pairwise_mean_product_matrix",
    "plot_narrative_timeseries",
    "plot_narrative_vs_events",
    "plot_narratives_ruptures",
    "procrustes_stiefel",
    "resample_constant_weekly",
    "resolve_storage_dir",
    "run_moving_tests_fdr",
    "save_and_download_figure",
    "short_label",
    "stiefel_optimize",
    "upper_triangle_values",
    "weekly_ema",
]
