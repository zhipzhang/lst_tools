"""Create an LST analysis workspace with linked data and DataCheck products."""

import argparse
import os
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from lst_tools.datacheck import (
    DataCheckTables,
    DataFilter,
    assign_zenith_bins,
    validate_zenith_bin_edges,
    zenith_bin_labels,
)
from lst_tools.dl3 import (
    DL3Product,
    discover_lst_dl3_products,
    load_dl3_requests,
    select_configured_dl3_products,
)
from lst_tools.helper import find_lst_data_path, glob_files

DATACHECK_DIR = (
    "/fefs/aswg/data/real/DL1/datacheck_files/night_wise/",
    "/fefs/onsite/data/lst-pipe/LSTN-01/DL1/datacheck_files/night_wise/",
)
SUPPORTED_LEVELS = ("dl1", "dl2")


def create_safe_link(source: Path, destination: Path) -> str:
    """Create a symlink without replacing an existing file or different link."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        if Path(os.path.realpath(destination)) == Path(os.path.realpath(source)):
            return "existing"
        warnings.warn(f"Link destination already points elsewhere: {destination}", stacklevel=2)
        return "conflict"

    if destination.exists():
        warnings.warn(f"Refusing to replace existing file: {destination}", stacklevel=2)
        return "conflict"

    destination.symlink_to(source)
    return "created"


def create_data_links(
    selected_runs: pd.DataFrame,
    output_root: Path,
    levels: tuple[str, ...],
    path_finder=find_lst_data_path,
) -> Counter:
    """Create idempotent DL1/DL2 links grouped by zenith-angle bin."""
    unsupported = set(levels).difference(SUPPORTED_LEVELS)
    if unsupported:
        raise ValueError(f"Unsupported data levels: {sorted(unsupported)}")

    counts = Counter()

    for _, row in selected_runs.iterrows():
        run_number = int(row["run_number"])
        zenith_bin = row["zenith_bin"]
        if not isinstance(zenith_bin, str):
            warnings.warn(f"Run {run_number} has no zenith bin; skipping links", stacklevel=2)
            counts["unbinned"] += 1
            continue

        date = int(row["date"])
        for level in levels:
            source_name = path_finder(date, int(run_number), level=level)
            if not source_name:
                counts[f"{level}:missing"] += 1
                continue

            source = Path(source_name)
            output_dir = output_root / level / zenith_bin
            destination = output_dir / source.name
            status = create_safe_link(source, destination)
            counts[f"{level}:{status}"] += 1

    return counts


def create_dl3_links(
    selected_runs: pd.DataFrame,
    output_root: Path,
    dl3_config: dict,
    product_finder=discover_lst_dl3_products,
) -> Counter:
    """Link only explicitly configured DL3 products for the selected runs."""
    requests = load_dl3_requests(dl3_config)
    counts = Counter()

    for _, row in selected_runs.iterrows():
        run_number = int(row["run_number"])
        zenith_bin = row["zenith_bin"]
        if not isinstance(zenith_bin, str):
            counts["dl3:unbinned"] += 1
            continue

        date = int(row["date"])
        available = product_finder(date, int(run_number))
        configured = select_configured_dl3_products(available, requests)

        for request in requests:
            product: DL3Product | None = configured.get(request)
            counter_prefix = f"dl3:{request.name}:{request.cut_config}"
            if product is None:
                counts[f"{counter_prefix}:missing"] += 1
                continue

            destination = output_root / "dl3" / request.name / request.cut_config / zenith_bin / product.path.name
            status = create_safe_link(product.path, destination)
            counts[f"{counter_prefix}:{status}"] += 1

    return counts


def prepare_working_directory(
    output_root: Path,
    edges: tuple[float, ...],
    initialization: dict,
    dl3_config: dict,
) -> tuple[str, ...]:
    """Create the configured DL1, DL2, DL3, and DataCheck directories."""
    labels = zenith_bin_labels(edges)
    requested_levels = tuple(level for level in SUPPORTED_LEVELS if initialization.get(level, False))

    for level in requested_levels:
        for label in labels:
            (output_root / level / label).mkdir(parents=True, exist_ok=True)

    if initialization.get("data_check", True):
        (output_root / "data_check").mkdir(parents=True, exist_ok=True)

    if dl3_config.get("enabled", False):
        for request in load_dl3_requests(dl3_config):
            for label in labels:
                (output_root / "dl3" / request.name / request.cut_config / label).mkdir(
                    parents=True,
                    exist_ok=True,
                )

    return requested_levels


def select_runs(
    tables: DataCheckTables,
    source_config: dict,
    filter_config: dict,
    edges: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, DataFilter]:
    """Calculate run statistics and apply the configured data-quality cuts."""
    cuts = dict(filter_config.get("cuts", {}))
    cuts["min_zenith_angle"] = edges[0]
    cuts["max_zenith_angle"] = edges[-1]
    data_filter = DataFilter(
        source_ra=source_config["ra"],
        source_dec=source_config["dec"],
        **cuts,
    )

    runs_after_basic_cuts = data_filter.apply_basic_cuts(tables.statistics)
    selected_runs = runs_after_basic_cuts
    if filter_config.get("with_advanced", False):
        selected_runs = data_filter.apply_advanced_cuts(selected_runs)

    return assign_zenith_bins(selected_runs, edges), runs_after_basic_cuts, data_filter


def save_datacheck_outputs(
    tables: DataCheckTables,
    selected_runs: pd.DataFrame,
    runs_after_basic_cuts: pd.DataFrame,
    data_filter: DataFilter,
    output_dir: Path,
    stem: str,
) -> None:
    """Save selected DataCheck tables and advanced-cut diagnostic plots."""
    import matplotlib.pyplot as plt

    from lst_tools.datacheck import plot_advanced_distributions

    run_numbers = selected_runs["run_number"].tolist()
    tables.select_runs(run_numbers).save_to_h5file(
        output_dir / f"data_check_{stem}.h5",
        overwrite=True,
    )

    figures = plot_advanced_distributions(runs_after_basic_cuts, data_filter)
    for name, figure in figures.items():
        figure.savefig(output_dir / f"advanced_cut_{name}.png")
        plt.close(figure)


def load_config(path: Path) -> dict:
    """Load an initialization configuration from TOML."""
    with path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def load_datacheck_tables() -> DataCheckTables:
    """Load all available nightly DataCheck files."""
    files = glob_files(DATACHECK_DIR, "DL1_datacheck_20*.h5")
    if not files:
        raise FileNotFoundError(f"No datacheck files found under {DATACHECK_DIR}")
    return DataCheckTables.from_files(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="TOML configuration file")
    parser.add_argument("-c", "--config", dest="config_option", type=Path, help="TOML configuration file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.cwd(),
        help="Analysis output directory (default: current directory)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_file = args.config_option or args.config_file
    if config_file is None:
        parser.error("a configuration file is required")

    config = load_config(config_file)
    source_config = config["source"]
    source_name = "_".join(source_config["name"].split())
    edges = validate_zenith_bin_edges(config["zenith_binning"]["edges_deg"])
    output_root = args.output.resolve()
    initialization = config.get("initialization", {})
    dl3_config = config.get("dl3", {})

    data_check_tables = load_datacheck_tables()
    selected_runs, runs_after_basic_cuts, data_filter = select_runs(
        data_check_tables,
        source_config,
        config.get("data_filter", {}),
        edges,
    )
    requested_levels = prepare_working_directory(output_root, edges, initialization, dl3_config)

    if initialization.get("data_check", True):
        stem = f"{source_name}_ra_{source_config['ra']}_dec_{source_config['dec']}"
        save_datacheck_outputs(
            data_check_tables,
            selected_runs,
            runs_after_basic_cuts,
            data_filter,
            output_root / "data_check",
            stem,
        )

    counts = create_data_links(selected_runs, output_root, requested_levels)
    if dl3_config.get("enabled", False):
        counts.update(create_dl3_links(selected_runs, output_root, dl3_config))

    print(f"Selected {len(selected_runs)} runs")
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
