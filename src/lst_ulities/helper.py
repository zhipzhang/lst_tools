import glob


def glob_files(path: str | list[str], pattern: str) -> list[str]:
    """
    Glob files in the given path using the given pattern.
    """
    if isinstance(path, str):
        files = glob.glob(f"{path}/{pattern}")
        return sorted(files)
    elif isinstance(path, list):
        files = []
        for p in path:
            files.extend(glob.glob(f"{p}/{pattern}"))
        return sorted(files)


def init_plot():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        }
    )


def plot_histogram(
    data,
    min=None,
    max=None,
    bins=None,
    ax=None,
    xlabel: str | None = None,
    title: str | None = None,
):
    """
    Plot a 1D histogram of ``data`` with optional ``min``/``max`` thresholds.

    Parameters
    ----------
    data : array-like
        Values to histogram.  pandas Series and numpy arrays are accepted.
    min : float, optional
        Lower threshold.  Values below ``min`` are counted as failing.
    max : float, optional
        Upper threshold.  Values above ``max`` are counted as failing.
    bins : int, optional
        Number of histogram bins.  Defaults to 30.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.  If None, a new figure and axes are created.
    xlabel : str, optional
        X-axis label.  Defaults to ``"Value"``.
    title : str, optional
        Axes title.  Defaults to ``"Histogram"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if bins is None:
        bins = 30

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    else:
        fig = ax.figure

    # Handle pandas-like input and drop NaNs
    if hasattr(data, "dropna"):
        values = data.dropna().to_numpy()
    else:
        values = np.asarray(data, dtype=float)
        values = values[~np.isnan(values)]

    n_total = len(values)

    ax.hist(
        values,
        bins=bins,
        color="0.25",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    # Thresholds and pass/fail counts
    if min is not None and max is not None:
        passed = ((values >= min) & (values <= max)).sum()
        ax.axvline(
            min,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"range: [{min:.2f}, {max:.2f}]",
        )
        ax.axvline(max, color="crimson", linestyle="--", linewidth=1.5)
    elif min is not None:
        passed = (values >= min).sum()
        ax.axvline(
            min,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"min ≥ {min:.2f}",
        )
    elif max is not None:
        passed = (values <= max).sum()
        ax.axvline(
            max,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"max ≤ {max:.2f}",
        )
    else:
        passed = n_total

    failed = n_total - passed

    ax.set_xlabel(xlabel if xlabel is not None else "Value")
    ax.set_ylabel("Number of Runs")
    ax.set_title(title if title is not None else "Histogram")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7, which="both")

    ax.text(
        0.97,
        0.97,
        f"Pass: {passed}/{n_total}\nFail: {failed}/{n_total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.8",
            alpha=0.9,
        ),
    )

    return fig, ax


POSSIBLE_PATH = ("/fefs/aswg/data/real", "/fefs/onsite/data/lst-pipe/LSTN-01")


def find_lst_data_path(date: int, run_number: int, level="dl1"):
    import warnings

    level = level.upper()
    if level not in ["DL1", "DL2"]:
        raise ValueError(f"Invalid level: {level}. Must be 'DL1' or 'DL2'.")
    file_name = f"{level.lower()}_LST-1.Run{run_number:05d}.h5"

    matched_files = []
    for path in POSSIBLE_PATH:
        prefix_path = f"{path}/{level}/{date}"
        matched_files.extend(glob.glob(f"{prefix_path}/**/{file_name}", recursive=True))

    if len(matched_files) == 0:
        warnings.warn(
            f"No file found for {file_name} under any of {POSSIBLE_PATH}",
            stacklevel=2,
        )
        raise FileNotFoundError(f"No file found for {file_name}")

    if len(matched_files) > 1:
        warnings.warn(
            f"Multiple files found for {file_name}: {matched_files}",
            stacklevel=2,
        )
        # Don't raise an error, just return the first match

    return matched_files[0]


def find_lst_subrun_files(date: int, run_number: int, level="dl1") -> list[str]:
    """Find subrun files for a given run, e.g. dl1_LST-1.Run20752.0044.h5.

    The glob pattern ``????.h5`` matches exactly 4-digit subrun suffixes,
    so plain run-summary files (e.g. dl1_LST-1.Run20752.h5) are never returned.

    Returns a sorted list of matching subrun file paths (may be empty).
    """
    import warnings

    level = level.upper()
    if level not in ["DL1", "DL2"]:
        raise ValueError(f"Invalid level: {level}. Must be 'DL1' or 'DL2'.")

    # ????.h5 matches exactly 4-character subrun numbers (e.g. 0044), not plain .h5
    pattern = f"{level.lower()}_LST-1.Run{run_number:05d}.????.h5"

    matched_files = []
    for path in POSSIBLE_PATH:
        prefix_path = f"{path}/{level}/{date}"
        matched_files.extend(glob.glob(f"{prefix_path}/**/{pattern}", recursive=True))

    if len(matched_files) == 0:
        warnings.warn(
            f"No subrun files found for Run{run_number:05d} under any of {POSSIBLE_PATH}",
            stacklevel=2,
        )

    return sorted(matched_files)
