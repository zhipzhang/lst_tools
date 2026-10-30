"""Run the LST analysis pipeline, from run selection to DL3.

The per-level logic lives in ``lst_tools.pipeline``: each data level has a
producer that either links files the standard pipeline already produced on
La Palma or builds them ourselves, selected by the config's ``source``
mode (``agent_doc/20261022_init_lstana_source_modes.md``):

1-2. select runs (nightly DataCheck statistics + configured quality cuts)
     and save the data-check tables and cut diagnostics
3.   DL1Producer: link DL1 (the external input data)
4.   DL2Producer: build DL2 from the linked DL1 with the configured RF
     models, or link the official DL2
5.   DL3Producer: build DL3 from our DL2 + our IRFs, or link existing DL3
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import astropy.units as u

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from lst_tools.catalog import select_runs_away_from_sources
from lst_tools.datacheck import (
    DataCheckTables,
    DataFilter,
    assign_zenith_bins,
    validate_zenith_bin_edges,
)
from lst_tools.finder.helper import filter_date
from lst_tools.helper import glob_files
from lst_tools.pipeline import (
    DL1Producer,
    DL2Producer,
    DL3Producer,
    DLProducer,
    PipelineConfig,
    RunSelection,
    SourceConfig,
)

DATACHECK_DIR = (
    "/fefs/aswg/data/real/DL1/datacheck_files/night_wise/",
    "/fefs/onsite/data/lst-pipe/LSTN-01/DL1/datacheck_files/night_wise/",
)
DATACHECK_PATTERN = "DL1_datacheck_20*.h5"

DEFAULT_MIN_DISTANCE_DEG = 3.0
DEFAULT_EXTENSION_FACTOR = 2.5


def load_config(path: Path) -> dict:
    """Load the raw pipeline configuration from TOML (a plain dict)."""
    with path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def load_datacheck_tables(raw_config: dict) -> DataCheckTables:
    """Load the nightly DataCheck files within the configured date range.

    The date cuts narrow the file list *before* loading: each datacheck
    file is only opened if its night lies within
    ``[data_filter.cuts] first_date/last_date``.
    """
    files = glob_files(DATACHECK_DIR, DATACHECK_PATTERN)
    cuts = raw_config.get("data_filter", {}).get("cuts", {})
    files = filter_date(files, int(cuts.get("first_date", 0)), int(cuts.get("last_date", 20999999)))
    if not files:
        raise FileNotFoundError(f"No datacheck files found under {DATACHECK_DIR} within the configured date range")
    return DataCheckTables.from_files(files)


def select_runs(raw_config: dict) -> RunSelection:
    """Stages 1-2 input: apply the data-quality cuts and assign zenith bins.

    With ``[data_filter] away_source = true`` the target-angle cuts are
    disabled and runs pointed near any catalog source are rejected instead,
    so the selection holds off runs without a separate ``prepare-offruns``
    workspace.
    """
    source_config = SourceConfig.parse(raw_config.get("source"))
    edges = validate_zenith_bin_edges(raw_config["zenith_binning"]["edges_deg"])
    data_filter_config = raw_config.get("data_filter", {})
    cuts = dict(data_filter_config.get("cuts", {}))
    cuts["min_zenith_angle"] = edges[0]
    cuts["max_zenith_angle"] = edges[-1]
    away_source = bool(data_filter_config.get("away_source", False))
    if away_source:
        # A pointing away from every catalog source cannot be inside the
        # target-angle range, and the target itself is in the catalogs.
        cuts["min_angle_to_source"] = 0.0
        cuts["max_angle_to_source"] = 180.0
    data_filter = DataFilter(source_ra=source_config.ra, source_dec=source_config.dec, **cuts)

    tables = load_datacheck_tables(raw_config)
    after_basic_cuts = data_filter.apply_basic_cuts(tables.statistics)
    selected = after_basic_cuts
    if data_filter_config.get("with_advanced", False):
        selected = data_filter.apply_advanced_cuts(selected)
    if away_source:
        from lst_tools.catalog import load_tev_sources

        selected = select_runs_away_from_sources(
            selected,
            load_tev_sources(),
            min_separation=float(data_filter_config.get("min_distance", DEFAULT_MIN_DISTANCE_DEG)) * u.deg,
            extension_factor=float(data_filter_config.get("extension_factor", DEFAULT_EXTENSION_FACTOR)),
        )

    return RunSelection(
        edges=edges,
        selected=assign_zenith_bins(selected, edges),
        tables=tables,
        after_basic_cuts=after_basic_cuts,
        data_filter=data_filter,
    )


def save_datacheck_outputs(selection: RunSelection, raw_config: dict, output_dir: Path) -> None:
    """Save the selected DataCheck tables and advanced-cut diagnostic plots."""
    import matplotlib.pyplot as plt

    from lst_tools.datacheck import plot_advanced_distributions

    source_config = SourceConfig.parse(raw_config.get("source"))
    stem = "_".join(source_config.name.split()) + f"_ra_{source_config.ra}_dec_{source_config.dec}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if selection.tables is None:
        raise ValueError("save_datacheck_outputs requires the datacheck tables from select_runs")

    run_numbers = selection.selected["run_number"].tolist()
    selection.tables.select_runs(run_numbers).save_to_h5file(
        output_dir / f"data_check_{stem}.h5",
        overwrite=True,
    )

    figures = plot_advanced_distributions(selection.after_basic_cuts, selection.data_filter)
    for name, figure in figures.items():
        figure.savefig(output_dir / f"advanced_cut_{name}.png")
        plt.close(figure)


def make_producers(pipeline_config: PipelineConfig, raw_config: dict, output_root: Path) -> list[DLProducer]:
    """The producer chain; each level finds its input in the previous one."""
    source_config = SourceConfig.parse(raw_config.get("source"))
    dl1 = DL1Producer(pipeline_config.dl1, output_root)
    dl2 = DL2Producer(pipeline_config.dl2, output_root, dl1_directory=dl1.output_dir)
    dl3 = DL3Producer(pipeline_config.dl3, output_root, source=source_config, dl2_directory=dl2.output_dir)
    return [dl1, dl2, dl3]


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

    raw_config = load_config(config_file)
    pipeline_config = PipelineConfig.parse(raw_config)
    output_root = args.output.resolve()

    selection = select_runs(raw_config)  # 1-2
    save_datacheck_outputs(selection, raw_config, output_root / "data_check")

    counts: Counter = Counter()
    for producer in make_producers(pipeline_config, raw_config, output_root):  # 3-5
        if not producer.config.enabled:
            continue
        counts += producer.produce(selection)

    print(f"Selected {len(list(selection))} runs")
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
