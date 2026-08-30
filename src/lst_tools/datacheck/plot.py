import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..helper import init_plot, plot_histogram
from .datafilter import DataFilter


def plot_advanced_distributions(
    statistics: pd.DataFrame,
    data_filter: DataFilter,
) -> dict[str, Figure]:
    """Create standalone figures for advanced-cut distributions.

    ``statistics`` is plotted exactly as supplied. Callers decide whether to
    apply basic or advanced cuts before calling this function.
    """
    required_columns = set(data_filter.ADVANCED_COLUMNS) | {"n_subruns"}
    if not required_columns.issubset(statistics.columns):
        raise ValueError("advanced-cut columns and n_subruns must exist in statistics")

    init_plot()
    p_value_in_sigma = (statistics["mean_fit_p_value"] - 0.5) * np.sqrt(12 * statistics["n_subruns"])
    distributions = {
        "mean_diffuse_nsb_std": (
            statistics["mean_diffuse_nsb_std"],
            None,
            data_filter.max_diffuse_nsb_std,
            "mean diffuse NSB std",
        ),
        "mean_intensity_threshold": (
            statistics["mean_intensity_threshold"],
            None,
            data_filter.max_intensity_at_half_peak_rate,
            "mean intensity threshold",
        ),
        "mean_fit_p_value": (
            p_value_in_sigma,
            data_filter.min_mean_fit_p,
            None,
            "mean fit p-value (#sigma)",
        ),
        "mean_index": (
            statistics["mean_index"],
            data_filter.min_drdi_index,
            data_filter.max_drdi_index,
            "mean DRDI index",
        ),
        "mean_R422": (
            statistics["mean_R422"],
            data_filter.min_drdi_at_422pe,
            None,
            "mean R422",
        ),
        "fraction_around_mode_R422": (
            statistics["fraction_around_mode_R422"],
            data_filter.min_fraction_around_mode,
            None,
            "fraction around mode R422",
        ),
    }

    figures = {}
    for name, (values, minimum, maximum, xlabel) in distributions.items():
        figure, _ = plot_histogram(
            values,
            min=minimum,
            max=maximum,
            xlabel=xlabel,
            title=name,
        )
        figures[name] = figure

    return figures
