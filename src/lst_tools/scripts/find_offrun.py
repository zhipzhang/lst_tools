"""Find good off-source runs and collect their datacheck and DL2 files."""

import argparse
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from lst_tools.datacheck import DataCheckTables, DataFilter, RunStatistics
from lst_tools.helper import find_lst_data_path, glob_files
from lst_tools.scripts.init_lstana import DATACHECK_DIR, create_safe_link


def create_dl2_links(
    selected_runs: RunStatistics,
    output_dir: Path,
    path_finder=find_lst_data_path,
) -> Counter:
    """Create idempotent DL2 links for the selected off-source runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    for run_number, row in selected_runs.df.iterrows():
        date = int(row["date"])
        source_name = path_finder(date, int(run_number), level="dl2")
        if not source_name:
            counts["dl2:missing"] += 1
            continue

        source = Path(source_name)
        status = create_safe_link(source, output_dir / source.name)
        counts[f"dl2:{status}"] += 1

    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="TOML configuration file")
    parser.add_argument("-c", "--config", dest="config_option", type=Path, help="TOML configuration file")
    parser.add_argument(
        "--off-datacheck",
        type=Path,
        required=True,
        help="Directory in which to save filtered off-run datacheck files",
    )
    parser.add_argument(
        "--off-dl2",
        type=Path,
        required=True,
        help="Directory in which to create links to selected off-run DL2 files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_file = args.config_option or args.config_file
    if config_file is None:
        parser.error("a configuration file is required")

    with config_file.open("rb") as file_handle:
        config = tomllib.load(file_handle)

    data_check_files = glob_files(DATACHECK_DIR, "DL1_datacheck_20*.h5")
    if not data_check_files:
        raise FileNotFoundError(f"No datacheck files found under {DATACHECK_DIR}")

    data_check_tables = DataCheckTables.from_files(data_check_files)
    run_statistics = RunStatistics.from_tables(data_check_tables)

    source_config = config["source"]
    basic_cuts = dict(config["data_filter"]["basic_cuts"])
    data_filter = DataFilter(
        source_ra=source_config["ra"],
        source_dec=source_config["dec"],
        **basic_cuts,
    )
    selected_runs = data_filter.filter_good_offruns(run_statistics, min_galactic_b=0)
    selected_tables = data_check_tables.select_runs(selected_runs.run_numbers)

    off_datacheck_dir = args.off_datacheck.resolve()
    off_datacheck_dir.mkdir(parents=True, exist_ok=True)
    selected_tables.save_to_h5file(
        off_datacheck_dir / "DL1_datacheck_offruns.h5",
        overwrite=True,
    )
    selected_runs.save_to_h5file(
        off_datacheck_dir / "selected_offruns.h5",
        overwrite=True,
    )

    counts = create_dl2_links(selected_runs, args.off_dl2.resolve())
    print(f"Selected {len(selected_runs.df)} good off-source runs")
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
