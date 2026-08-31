"""Select catalog-separated off runs and prepare their DataCheck and DL2 files."""

import argparse
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import pandas as pd

from lst_tools.catalog import (
    CatalogSource,
    select_runs_away_from_sources,
)
from lst_tools.datacheck import DataCheckTables, DataFilter
from lst_tools.helper import find_lst_data_path
from lst_tools.scripts.init_lstana import create_safe_link, load_config, load_datacheck_tables


def select_offruns(
    statistics: pd.DataFrame,
    data_filter: DataFilter,
    sources: Iterable[CatalogSource],
    *,
    min_separation: u.Quantity = 3 * u.deg,
    extension_factor: float = 2.5,
) -> pd.DataFrame:
    """Apply quality cuts, then retain runs away from catalog sources.

    The source-angle cut in ``data_filter`` is disabled by replacing its
    bounds with the full 0–180 degree range. All other basic cuts and every
    advanced cut remain active.
    """
    quality_filter = replace(
        data_filter,
        min_angle_to_source=0,
        max_angle_to_source=180,
    )
    quality_run_numbers = quality_filter(statistics, advanced_cuts=True)
    quality_runs = statistics.loc[statistics["run_number"].isin(quality_run_numbers)]

    return select_runs_away_from_sources(
        quality_runs,
        sources,
        min_separation=min_separation,
        extension_factor=extension_factor,
    )


def create_dl2_links(
    selected_runs: pd.DataFrame,
    output_dir: Path,
    path_finder=find_lst_data_path,
) -> Counter:
    """Create idempotent DL2 links for the selected off runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    for row in selected_runs.itertuples(index=False):
        source_name = path_finder(int(row.date), int(row.run_number), level="dl2")
        if not source_name:
            counts["dl2:missing"] += 1
            continue

        source = Path(source_name)
        status = create_safe_link(source, output_dir / source.name)
        counts[f"dl2:{status}"] += 1

    return counts


def load_offrun_sources() -> tuple[CatalogSource, ...]:
    """Load the TeV catalogs used to reject source-contaminated pointings."""
    from lst_tools.catalog import load_hawc_sources, load_hess_sources, load_lhaaso_sources

    return load_hess_sources() + load_lhaaso_sources() + load_hawc_sources()


def save_datacheck_file(
    tables: DataCheckTables,
    selected_runs: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Save the selected runs as one DataCheck HDF5 file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "DL1_datacheck_offruns.h5"
    tables.select_runs(selected_runs["run_number"].tolist()).save_to_h5file(
        output_file,
        overwrite=True,
    )
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="TOML configuration file")
    parser.add_argument("-c", "--config", dest="config_option", type=Path, help="TOML configuration file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for the off-run workspace")
    parser.add_argument(
        "--min-separation",
        type=float,
        default=3.0,
        metavar="DEG",
        help="Minimum pointing separation from a point-like source in degrees (default: 3)",
    )
    parser.add_argument(
        "--extension-factor",
        type=float,
        default=2.5,
        help="Multiplier applied to catalog source extensions (default: 2.5)",
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
    cuts = dict(config.get("data_filter", {}).get("cuts", {}))
    data_filter = DataFilter(
        source_ra=source_config["ra"],
        source_dec=source_config["dec"],
        **cuts,
    )

    tables = load_datacheck_tables()
    selected_runs = select_offruns(
        tables.statistics,
        data_filter,
        load_offrun_sources(),
        min_separation=args.min_separation * u.deg,
        extension_factor=args.extension_factor,
    )

    output_root = args.output.resolve()
    datacheck_file = save_datacheck_file(tables, selected_runs, output_root / "data_check")
    counts = create_dl2_links(selected_runs, output_root / "dl2")

    print(f"Selected {len(selected_runs)} off runs")
    print(f"DataCheck: {datacheck_file}")
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
