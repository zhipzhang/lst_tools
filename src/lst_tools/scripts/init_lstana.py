"""Run the LST analysis pipeline, from run selection to DL3.

The six stages follow ``agent_doc/20261006_init_lstana_redesign.md``:

1. select runs (nightly DataCheck statistics + configured quality cuts)
2. data check (selected tables + cut diagnostics)
3. link DL1 (the external input data)
4. build DL2 (our own reconstruction with the configured RF models)
5. build IRFs (from the MC test dataset matching each DL2 file's provenance)
6. reduce to DL3 (our DL2 + our IRFs, via lstchain's DataReductionFITSWriter)

A stage runs when its config section is present (``[dl2]``, ``[irf]``) or
enabled (``[dl3]``); stages 1-3 always run.
"""

import argparse
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
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
from lst_tools.dl1.dl1event import dl1_to_dl2_batch
from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.dl3 import DL3Reducer
from lst_tools.event_filter import EventFilter
from lst_tools.helper import find_lst_data_path, glob_files
from lst_tools.irf import IRFConfig, IRFGenerator
from lst_tools.irf.utils import find_irf_nodes

DATACHECK_DIR = (
    "/fefs/aswg/data/real/DL1/datacheck_files/night_wise/",
    "/fefs/onsite/data/lst-pipe/LSTN-01/DL1/datacheck_files/night_wise/",
)
SUPPORTED_LEVELS = ("dl1", "dl2")

# The RF model directory must carry the MC group (NSB tuning, declination
# line) in its path: stages 5 and 6 parse it back from the DL2 provenance.
_NSB_TUNING_PATTERN = re.compile(r"nsb_tuning_[\d.]+")
_DEC_LINE_PATTERN = re.compile(r"dec_(?:min_)?\d+")


@dataclass
class RunSelection:
    """Stage-1 result: the selected runs and what the later stages need."""

    tables: DataCheckTables
    selected: pd.DataFrame  # with the zenith_bin column
    after_basic_cuts: pd.DataFrame
    data_filter: DataFilter
    edges: tuple[float, ...]


def load_config(path: Path) -> dict:
    """Load a pipeline configuration from TOML."""
    with path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def load_irf_config(config: dict) -> IRFConfig:
    """Build the IRF generation/selection config from the ``[irf]`` section."""
    section = config.get("irf")
    if section is None:
        raise ValueError(
            "An [irf] section with intensity_cut, leakage_cut and gh_cut or gh_efficiency is required"
        )
    event_filter = EventFilter.from_cuts(
        intensity_cut=section["intensity_cut"],
        leakage_cut=section["leakage_cut"],
        gh_cut=section.get("gh_cut"),
        gh_efficiency=section.get("gh_efficiency"),
    )
    return IRFConfig(event_filter)


def validate_config(config: dict) -> None:
    """Reject malformed stage configs before any stage runs."""
    if "dl2" in config:
        rf_directory = config["dl2"].get("rf_directory")
        if rf_directory is None:
            raise ValueError("[dl2] requires rf_directory (the RF model directory used to build DL2)")
        if not (_NSB_TUNING_PATTERN.search(rf_directory) and _DEC_LINE_PATTERN.search(rf_directory)):
            raise ValueError(
                f"[dl2] rf_directory must follow the nsb_tuning_<x>/dec_<NNNN> naming, got {rf_directory!r};"
                " the IRF and DL3 stages parse the MC group from it"
            )

    if config.get("dl3", {}).get("enabled", False) and "irf" not in config:
        raise ValueError("[dl3] enabled requires an [irf] section: the DL3 reduction shares its event selection")
    if "irf" in config:
        load_irf_config(config)  # raises on a malformed section


def load_datacheck_tables() -> DataCheckTables:
    """Load all available nightly DataCheck files."""
    files = glob_files(DATACHECK_DIR, "DL1_datacheck_20*.h5")
    if not files:
        raise FileNotFoundError(f"No datacheck files found under {DATACHECK_DIR}")
    return DataCheckTables.from_files(files)


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
    """Create idempotent data links grouped by zenith-angle bin."""
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


def prepare_working_directory(output_root: Path, edges: tuple[float, ...], config: dict) -> None:
    """Create the directory tree the pipeline stages write into."""
    labels = zenith_bin_labels(edges)
    (output_root / "data_check").mkdir(parents=True, exist_ok=True)
    for label in labels:
        (output_root / "dl1" / label).mkdir(parents=True, exist_ok=True)
    if "dl2" in config:
        for label in labels:
            (output_root / "dl2" / label).mkdir(parents=True, exist_ok=True)
    if config.get("dl3", {}).get("enabled", False):
        for label in labels:
            (output_root / "dl3" / label).mkdir(parents=True, exist_ok=True)


def select_runs(config: dict) -> RunSelection:
    """Stage 1: apply the data-quality cuts and assign zenith-angle bins."""
    source = config["source"]
    edges = validate_zenith_bin_edges(config["zenith_binning"]["edges_deg"])
    filter_config = config.get("data_filter", {})
    cuts = dict(filter_config.get("cuts", {}))
    cuts["min_zenith_angle"] = edges[0]
    cuts["max_zenith_angle"] = edges[-1]
    data_filter = DataFilter(source_ra=source["ra"], source_dec=source["dec"], **cuts)

    tables = load_datacheck_tables()
    after_basic_cuts = data_filter.apply_basic_cuts(tables.statistics)
    selected = after_basic_cuts
    if filter_config.get("with_advanced", False):
        selected = data_filter.apply_advanced_cuts(selected)

    return RunSelection(
        tables=tables,
        selected=assign_zenith_bins(selected, edges),
        after_basic_cuts=after_basic_cuts,
        data_filter=data_filter,
        edges=edges,
    )


def save_data_check(runs: RunSelection, config: dict, output_root: Path) -> None:
    """Stage 2: save the selected DataCheck tables and cut diagnostics."""
    source = config["source"]
    stem = "_".join(source["name"].split()) + f"_ra_{source['ra']}_dec_{source['dec']}"
    save_datacheck_outputs(
        runs.tables,
        runs.selected,
        runs.after_basic_cuts,
        runs.data_filter,
        output_root / "data_check",
        stem,
    )


def link_dl1(runs: RunSelection, output_root: Path) -> Counter:
    """Stage 3: link the selected DL1 files into the zenith-bin tree."""
    return create_data_links(runs.selected, output_root, ("dl1",))


def build_dl2(config: dict, output_root: Path) -> Counter:
    """Stage 4: reconstruct our DL2 from the linked DL1 with the RF models.

    Runs whose DL2 already exists are kept unless ``[dl2] overwrite = true``.
    """
    dl2_config = config.get("dl2")
    if dl2_config is None:
        return Counter()
    rf_directory = Path(dl2_config["rf_directory"]).expanduser()
    overwrite = dl2_config.get("overwrite", False)

    counts = Counter()
    for dl1_dir in sorted((output_root / "dl1").iterdir()):
        if not dl1_dir.is_dir():
            continue
        output_dir = output_root / "dl2" / dl1_dir.name
        dl1_files = sorted(dl1_dir.glob("dl1_*.h5"))
        pending = [
            dl1_file
            for dl1_file in dl1_files
            if overwrite or not (output_dir / dl1_file.name.replace("dl1", "dl2", 1)).exists()
        ]
        counts["dl2:existing"] += len(dl1_files) - len(pending)
        if not pending:
            continue

        dl1_to_dl2_batch(pending, output_dir, rf_directory, overwrite=overwrite)
        for dl1_file in pending:
            if (output_dir / dl1_file.name.replace("dl1", "dl2", 1)).exists():
                counts["dl2:built"] += 1
            else:
                warnings.warn(f"DL1ToDL2Tool did not create the DL2 file for {dl1_file}", stacklevel=2)
                counts["dl2:failed"] += 1

    return counts


def build_irfs(config: dict, output_root: Path) -> Counter:
    """Stage 5: one IRF per MC node of each DL2 file's MC group.

    The MC test dataset is located from each DL2 file's provenance (NSB
    tuning, declination line); the generation parameters come from ``[irf]``.
    """
    if "irf" not in config:
        return Counter()
    dl2_root = output_root / "dl2"
    if not dl2_root.is_dir():
        warnings.warn(f"No DL2 directory at {dl2_root}; skipping IRF generation", stacklevel=2)
        return Counter()

    generator = IRFGenerator(output_root / "irfs", load_irf_config(config))
    nodes = {}
    for dl2_file in sorted(dl2_root.glob("*/*.h5")):
        for node in find_irf_nodes(LSTDL2EventTable(dl2_file)):
            nodes.setdefault(node.relative_path, node)

    for node in nodes.values():
        generator.make_irf_node(node)
    return Counter({"irf:nodes": len(nodes)})


def reduce_to_dl3(config: dict, output_root: Path) -> Counter:
    """Stage 6: reduce our DL2 files to DL3 observations, per zenith bin."""
    dl3_config = config.get("dl3", {})
    if not dl3_config.get("enabled", False):
        return Counter()
    dl2_root = output_root / "dl2"
    if not dl2_root.is_dir():
        warnings.warn(f"No DL2 directory at {dl2_root}; skipping DL3 reduction", stacklevel=2)
        return Counter()

    source = config["source"]
    reducer = DL3Reducer(
        irfs_root=dl3_config.get("irf_directory", output_root / "irfs"),
        source_name=source["name"],
        source_ra=source["ra"],
        source_dec=source["dec"],
        irf_config=load_irf_config(config),
        interp_method=dl3_config.get("interp_method", "linear"),
        use_nearest_irf_node=dl3_config.get("use_nearest_irf_node", False),
        overwrite=dl3_config.get("overwrite", False),
    )

    counts = Counter()
    for dl2_dir in sorted(dl2_root.iterdir()):
        if not dl2_dir.is_dir():
            continue
        dl2_files = sorted(dl2_dir.glob("dl2_*.h5"))
        if not dl2_files:
            counts["dl3:no_dl2"] += 1
            continue

        reduced = reducer.reduce(dl2_files, output_root / "dl3" / dl2_dir.name)
        counts["dl3:reduced"] += len(reduced)
        counts["dl3:failed"] += len(dl2_files) - len(reduced)

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_file = args.config_option or args.config_file
    if config_file is None:
        parser.error("a configuration file is required")

    config = load_config(config_file)
    validate_config(config)
    output_root = args.output.resolve()

    runs = select_runs(config)  # 1. select runs
    prepare_working_directory(output_root, runs.edges, config)
    save_data_check(runs, config, output_root)  # 2. data check
    counts = link_dl1(runs, output_root)  # 3. link DL1
    counts += build_dl2(config, output_root)  # 4. build DL2
    counts += build_irfs(config, output_root)  # 5. build IRFs
    counts += reduce_to_dl3(config, output_root)  # 6. reduce to DL3

    print(f"Selected {len(runs.selected)} runs")
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
