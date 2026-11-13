"""
I/O and storage resolution utilities:
- Auto-detects Google Colab / Google Drive vs. local `resources/` folder
- Finds the freshest dated file matching a pattern (e.g. `cnn_news_YYYY-MM-DD.json`)
- Saves figures to disk and optionally triggers `google.colab.files.download`
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def get_repo_root() -> Path:
    """Return the root directory of the narrative-geometry repository."""
    return Path(__file__).resolve().parent.parent


def get_resources_dir() -> Path:
    """Return the path to `<repo_root>/resources`, creating it if needed."""
    res_dir = get_repo_root() / "resources"
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def resolve_storage_dir(
    drive_subdir: str = "data_for_colab",
    mount_point: str = "/content/drive",
    local_dir: str | Path | None = None,
) -> Path:
    """
    Resolve the storage directory for intermediate/cached data files:
    1. If running in Google Colab, attempts to mount Google Drive at `mount_point`
       and uses `<mount_point>/My Drive/<drive_subdir>` if available.
    2. If `<mount_point>/My Drive/<drive_subdir>` already exists on the filesystem,
       uses it directly.
    3. Otherwise falls back to `local_dir` (defaults to `<repo_root>/resources`).
    """
    if local_dir is None:
        fallback = get_resources_dir()
    else:
        fallback = Path(local_dir).resolve()

    drive_target = Path(mount_point) / "My Drive" / drive_subdir

    try:
        from google.colab import drive  # type: ignore

        if not Path(mount_point).is_dir() or not (Path(mount_point) / "My Drive").exists():
            drive.mount(mount_point)
        drive_target.mkdir(parents=True, exist_ok=True)
        print(f"[storage] Using Google Drive directory: {drive_target}")
        return drive_target
    except Exception:
        if drive_target.is_dir():
            print(f"[storage] Using existing Google Drive path: {drive_target}")
            return drive_target

    fallback.mkdir(parents=True, exist_ok=True)
    print(f"[storage] Google Drive not detected; using local directory: {fallback}")
    return fallback


def find_freshest_dated_file(
    directory: str | Path,
    prefix: str = "cnn_news_",
    suffix: str = ".json",
    fallback_dirs: list[str | Path] | None = None,
) -> Path | None:
    """
    Scan `directory` (and optional `fallback_dirs`) for files matching
    `<prefix>YYYY-MM-DD<suffix>` and return the Path with the most recent date.
    """
    search_dirs = [Path(directory)]
    if fallback_dirs:
        for d in fallback_dirs:
            p = Path(d)
            if p not in search_dirs:
                search_dirs.append(p)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(suffix)}$")

    most_recent_path: Path | None = None
    most_recent_date: datetime | None = None

    for d in search_dirs:
        if not d.is_dir():
            continue
        for filename in os.listdir(d):
            match = pattern.match(filename)
            if match:
                date_str = match.group(1)
                try:
                    current_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue
                if most_recent_date is None or current_date > most_recent_date:
                    most_recent_date = current_date
                    most_recent_path = d / filename

    return most_recent_path


def save_and_download_figure(
    basename: str,
    exts: tuple[str, ...] = (".png", ".pdf"),
    dpi: int = 150,
    bbox_inches: str = "tight",
    fig=None,
) -> list[Path]:
    """
    Save the current Matplotlib figure (or `fig`) under `basename + ext` for each
    extension in `exts`. If running in Google Colab, also triggers browser download.
    """
    saved_paths: list[Path] = []
    target_fig = fig if fig is not None else plt

    try:
        from google.colab import files as colab_files  # type: ignore
    except Exception:
        colab_files = None

    for ext in exts:
        fname = f"{basename}{ext}"
        target_fig.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
        saved_paths.append(Path(fname))
        if colab_files is not None:
            try:
                colab_files.download(fname)
            except Exception:
                pass

    return saved_paths
