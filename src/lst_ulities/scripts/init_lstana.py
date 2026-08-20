"""Initialize a run-wise LST analysis split into zenith-angle bins."""

import argparse
import os
import warnings
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from lst_ulities.datacheck import (
    DataCheckTables,
    DataFilter,
    RunStatistics,
    validate_zenith_bin_edges,
    zenith_bin_labels,
)
from lst_ulities.dl3 import (
    DL3Product,
    discover_lst_dl3_products,
    load_dl3_requests,
    select_configured_dl3_products,
)
from lst_ulities.helper import find_lst_data_path, glob_files

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
    selected_runs: RunStatistics,
    output_root: Path,
    levels: tuple[str, ...],
    path_finder=find_lst_data_path,
) -> Counter:
    """Create idempotent DL1/DL2 links grouped by zenith-angle bin."""
    unsupported = set(levels).difference(SUPPORTED_LEVELS)
    if unsupported:
        raise ValueError(f"Unsupported data levels: {sorted(unsupported)}")

    counts = Counter()

    for run_number, row in selected_runs.df.iterrows():
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
    selected_runs: RunStatistics,
    output_root: Path,
    dl3_config: dict,
    product_finder=discover_lst_dl3_products,
) -> Counter:
    """Link only explicitly configured DL3 products for the selected runs."""
    requests = load_dl3_requests(dl3_config)
    counts = Counter()

    for run_number, row in selected_runs.df.iterrows():
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    config_file = args.config_option or args.config_file
    if config_file is None:
        parser.error("a configuration file is required")

    with config_file.open("rb") as file_handle:
        config = tomllib.load(file_handle)

    source_ra = config["source"]["ra"]
    source_dec = config["source"]["dec"]
    source_name = "_".join(config["source"]["name"].split())
    edges = validate_zenith_bin_edges(config["zenith_binning"]["edges_deg"])

    initialization = config["initialization"]
    requested_levels = tuple(level for level in SUPPORTED_LEVELS if initialization.get(level, False))
    output_root = args.output.resolve()
    for level in requested_levels:
        for label in zenith_bin_labels(edges):
            (output_root / level / label).mkdir(parents=True, exist_ok=True)

    data_check_files = glob_files(DATACHECK_DIR, "DL1_datacheck_20*.h5")
    if not data_check_files:
        raise FileNotFoundError(f"No datacheck files found under {DATACHECK_DIR}")

    data_check_tables = DataCheckTables.from_files(data_check_files)
    run_statistics = RunStatistics.from_tables(data_check_tables)

    basic_cuts = dict(config["data_filter"]["basic_cuts"])
    basic_cuts["min_zenith_angle"] = edges[0]
    basic_cuts["max_zenith_angle"] = edges[-1]
    data_filter = DataFilter(source_ra=source_ra, source_dec=source_dec, **basic_cuts)
    use_advanced_cuts = config["data_filter"].get("with_advanced", False)

    selected_runs = data_filter(run_statistics, advanced_cuts=use_advanced_cuts)
    selected_runs = selected_runs.assign_zenith_bins(edges)

    if initialization.get("data_check", True):
        data_check_dir = output_root / "data_check"
        data_check_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{source_name}_ra_{source_ra}_dec_{source_dec}"
        selected_tables = data_check_tables.select_runs(selected_runs.run_numbers)
        selected_tables.save_to_h5file(data_check_dir / f"data_check_{stem}.h5", overwrite=True)
        selected_runs.save_to_h5file(data_check_dir / f"selected_runs_{stem}.h5", overwrite=True)

    counts = create_data_links(selected_runs, output_root, requested_levels)
    dl3_config = config.get("dl3", {})
    if dl3_config.get("enabled", False):
        counts.update(create_dl3_links(selected_runs, output_root, dl3_config))
    print(f"Selected {len(selected_runs.df)} runs")
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
