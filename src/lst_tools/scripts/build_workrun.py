"""Build an analysis workspace for one LST DL2 run.

The command locates one target DL2 file, selects compatible off runs from
DataCheck statistics, reconstructs their DL1 files with the target's trained
models, prepares the matching IRFs, and records the result in TOML.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from ctapipe.core import Tool, ToolConfigurationError, traits
from lstchain.tools.lstchain_dl1_to_dl2 import DL1ToDL2Tool

from lst_tools.datacheck import DataCheckTables
from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.irf import IRFGenerator
from lst_tools.irf.utils import MC_DL2_PATH, find_dl2_mc_path
from lst_tools.scripts.init_lstana import create_safe_link

_RUN_PATTERN = re.compile(r"Run(?P<run_number>\d+)")
_TAILCUT_PATTERN = re.compile(r"(?:^|[/_])tailcut(?P<levels>\d+)(?:[/_]|$)")


@dataclass(frozen=True)
class OffRunSelectionBounds:
    """Bounds used to select off runs similar to the target run."""

    nsb_min: float
    nsb_max: float
    zenith_min_deg: float
    zenith_max_deg: float
    month_day_min: int
    month_day_max: int


def find_tailcuts_level(path: str | Path) -> tuple[int, int] | None:
    """Extract picture and boundary tailcut levels from a resolved path."""
    match = _TAILCUT_PATTERN.search(str(path))
    if match is None:
        return None

    digits = match.group("levels")
    if len(digits) % 2:
        return None

    midpoint = len(digits) // 2
    return int(digits[:midpoint]), int(digits[midpoint:])


def find_target_dl2_file(dl2_path: str | Path, run_number: int) -> Path:
    """Find exactly one DL2 file for ``run_number`` under a file or directory."""
    search_path = Path(dl2_path).expanduser()
    candidates = [search_path] if search_path.is_file() else list(search_path.rglob("*.h5"))
    matches = []
    for candidate in candidates:
        run_match = _RUN_PATTERN.search(candidate.name)
        if "dl2" in candidate.name.lower() and run_match and int(run_match.group("run_number")) == run_number:
            matches.append(candidate)

    if not matches:
        raise ToolConfigurationError(f"No DL2 file found for Run{run_number:05d} under {search_path}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in sorted(matches))
        raise ToolConfigurationError(f"Multiple DL2 files found for Run{run_number:05d}: {paths}")
    return matches[0].resolve()


def find_datacheck_files(data_check_path: str | Path) -> list[Path]:
    """Return the DataCheck HDF5 files contained in a file or directory."""
    search_path = Path(data_check_path).expanduser()
    files = [search_path] if search_path.is_file() else sorted(search_path.rglob("*.h5"))
    if not files:
        raise ToolConfigurationError(f"No DataCheck files found under {search_path}")
    return [path.resolve() for path in files]


def select_matching_off_runs(
    statistics: pd.DataFrame,
    run_number: int,
    *,
    nsb_relative_tolerance: float,
    zenith_tolerance_deg: float,
    month_day_tolerance: int,
) -> tuple[pd.DataFrame, OffRunSelectionBounds]:
    """Select runs within the configured NSB, zenith, and calendar bounds."""
    required_columns = {"run_number", "date", "mean_cos_zd", "mean_diffuse_nsb_std"}
    missing_columns = required_columns.difference(statistics.columns)
    if missing_columns:
        raise ValueError(f"DataCheck statistics are missing columns: {sorted(missing_columns)}")

    target_rows = statistics.loc[statistics["run_number"].eq(run_number)]
    if target_rows.empty:
        raise ToolConfigurationError(f"Run{run_number:05d} is missing from the DataCheck statistics")
    if len(target_rows) > 1:
        raise ToolConfigurationError(f"Run{run_number:05d} occurs more than once in the DataCheck statistics")

    target = target_rows.iloc[0]
    target_nsb = float(target["mean_diffuse_nsb_std"])
    target_cos_zenith = float(np.clip(target["mean_cos_zd"], -1, 1))
    if not np.isfinite(target_nsb) or not np.isfinite(target_cos_zenith):
        raise ToolConfigurationError(f"Run{run_number:05d} has invalid NSB or zenith statistics")
    target_zenith_deg = float(np.degrees(np.arccos(target_cos_zenith)))
    target_month_day = int(target["date"]) % 10000

    bounds = OffRunSelectionBounds(
        nsb_min=target_nsb * (1 - nsb_relative_tolerance),
        nsb_max=target_nsb * (1 + nsb_relative_tolerance),
        zenith_min_deg=max(0.0, target_zenith_deg - zenith_tolerance_deg),
        zenith_max_deg=min(180.0, target_zenith_deg + zenith_tolerance_deg),
        month_day_min=target_month_day - month_day_tolerance,
        month_day_max=target_month_day + month_day_tolerance,
    )

    zenith_deg = np.degrees(np.arccos(statistics["mean_cos_zd"].clip(-1, 1)))
    month_day = statistics["date"].astype(int) % 10000
    selection_mask = (
        statistics["mean_diffuse_nsb_std"].between(bounds.nsb_min, bounds.nsb_max)
        & zenith_deg.between(bounds.zenith_min_deg, bounds.zenith_max_deg)
        & month_day.between(bounds.month_day_min, bounds.month_day_max)
        & statistics["run_number"].ne(run_number)
    )
    return statistics.loc[selection_mask].copy(), bounds


def find_compatible_offrun_dl1_files(
    offrun_dl1_path: str | Path,
    run_numbers: list[int],
    expected_tailcuts: tuple[int, int],
) -> list[Path]:
    """Find off-run DL1 files whose resolved paths have matching tailcuts."""
    search_path = Path(offrun_dl1_path).expanduser()
    compatible_files: list[Path] = []
    seen_sources: set[Path] = set()

    for run_number in run_numbers:
        matching_files = sorted(search_path.rglob(f"*Run{run_number:05d}*.h5"))
        matching_files = [path for path in matching_files if "dl1" in path.name.lower()]
        if not matching_files:
            warnings.warn(
                f"No DL1 file found for selected off run Run{run_number:05d} under {search_path}; skipping",
                stacklevel=2,
            )
            continue

        for linked_path in matching_files:
            try:
                source_path = linked_path.resolve(strict=True)
            except FileNotFoundError:
                warnings.warn(f"Broken DL1 link {linked_path}; skipping", stacklevel=2)
                continue

            tailcuts = find_tailcuts_level(source_path)
            if tailcuts != expected_tailcuts:
                warnings.warn(
                    f"Tailcuts mismatch for {linked_path}: found {tailcuts}, expected {expected_tailcuts}; skipping",
                    stacklevel=2,
                )
                continue
            if source_path not in seen_sources:
                compatible_files.append(source_path)
                seen_sources.add(source_path)

    return compatible_files


def expected_dl2_path(dl1_path: Path, output_dir: Path) -> Path:
    """Return the output filename used by ``DL1ToDL2Tool``."""
    return output_dir / dl1_path.name.replace("dl1", "dl2", 1)


def reconstruct_off_runs(
    dl1_files: list[Path],
    model_directory: Path,
    output_dir: Path,
    *,
    parent: Tool | None = None,
) -> list[Path]:
    """Create missing off-run DL2 files using the target run's models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_inputs = [path for path in dl1_files if not expected_dl2_path(path, output_dir).exists()]

    if pending_inputs:
        reconstruction_tool = DL1ToDL2Tool(
            parent=parent,
            input_files=pending_inputs,
            path_models=model_directory,
            output_dir=output_dir,
        )
        reconstruction_tool.setup()
        reconstruction_tool.start()

    expected_outputs = [expected_dl2_path(dl1, output_dir) for dl1 in dl1_files]
    for missing_output in (path for path in expected_outputs if not path.exists()):
        warnings.warn(f"DL1ToDL2Tool did not create {missing_output}", stacklevel=2)
    return [path for path in expected_outputs if path.exists()]


def _toml_value(value: object) -> str:
    """Serialize the scalar/list values used by the work-run configuration."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value).__name__}")


def write_workrun_config(path: Path, sections: dict[str, dict[str, object]]) -> None:
    """Write the important inputs, selection bounds, and products as TOML."""
    lines = ["# Generated by build-workrun", ""]
    for section_name, values in sections.items():
        lines.append(f"[{section_name}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items() if value is not None)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


class BuildWorkRun(Tool):
    """Build the DL2, off-run, and IRF products for one observation run."""

    name = "build-workrun"
    description = __doc__

    run_number = traits.Int(default_value=None, allow_none=True, help="Run number to build").tag(config=True)
    dl2_path = traits.Path(default_value="./dl2", help="Target DL2 file or directory").tag(config=True)
    data_check_path = traits.Path(
        default_value="./data_check",
        help="DataCheck HDF5 file or directory",
    ).tag(config=True)
    offrun_dl1_path = traits.Path(
        default_value="./offruns/dl1",
        help="Directory containing candidate off-run DL1 files or links",
    ).tag(config=True)
    mc_dl2_path = traits.Path(
        default_value=MC_DL2_PATH,
        help="AllSky MC DL2 root used to locate IRF nodes",
    ).tag(config=True)
    irf_output_dir = traits.Path(default_value="./irf", help="Shared IRF output directory").tag(config=True)
    output_dir = traits.Path(default_value="./workdir", help="Root directory for RunXXXXX workspaces").tag(config=True)
    zenith_tolerance_deg = traits.Float(
        default_value=2.0,
        help="Maximum zenith-angle difference for off runs, in degrees",
    ).tag(config=True)
    nsb_relative_tolerance = traits.Float(
        default_value=0.1,
        help="Maximum fractional NSB difference for off runs",
    ).tag(config=True)
    month_day_tolerance = traits.Int(
        default_value=300,
        help="Allowed numeric MMDD difference for off runs",
    ).tag(config=True)
    gh_efficiency = traits.Float(default_value=0.7, help="Gamma efficiency used for IRF generation").tag(config=True)

    aliases: ClassVar[dict[tuple[str, ...], str]] = {
        ("r", "run", "run-number", "runnumber"): "BuildWorkRun.run_number",
        ("dl2", "dl2-path"): "BuildWorkRun.dl2_path",
        ("data-check", "data-check-path"): "BuildWorkRun.data_check_path",
        ("offrun-dl1", "offrun-dl1-path"): "BuildWorkRun.offrun_dl1_path",
        ("mc-dl2", "mc-dl2-path"): "BuildWorkRun.mc_dl2_path",
        ("irf-output", "irf-output-dir"): "BuildWorkRun.irf_output_dir",
        ("o", "output", "output-dir"): "BuildWorkRun.output_dir",
        ("zenith-off", "zenith-tolerance"): "BuildWorkRun.zenith_tolerance_deg",
        ("nsb-level-off", "nsb-tolerance"): "BuildWorkRun.nsb_relative_tolerance",
        ("month-day-tolerance",): "BuildWorkRun.month_day_tolerance",
        ("gh-efficiency",): "BuildWorkRun.gh_efficiency",
    }

    def setup(self) -> None:
        if self.run_number is None or self.run_number < 0:
            raise ToolConfigurationError("run_number is required and must be non-negative")
        if not 0 <= self.nsb_relative_tolerance <= 1:
            raise ToolConfigurationError("nsb_relative_tolerance must be between 0 and 1")
        if self.zenith_tolerance_deg < 0:
            raise ToolConfigurationError("zenith_tolerance_deg must be non-negative")
        if self.month_day_tolerance < 0:
            raise ToolConfigurationError("month_day_tolerance must be non-negative")
        if not 0 < self.gh_efficiency <= 1:
            raise ToolConfigurationError("gh_efficiency must be in the interval (0, 1]")

        self.target_dl2_file = find_target_dl2_file(self.dl2_path, self.run_number)
        self.target_dl2_table = LSTDL2EventTable(self.target_dl2_file)
        if self.target_dl2_table.model_directory is None:
            raise ToolConfigurationError(f"No trained-model directory found in {self.target_dl2_file} provenance")
        if self.target_dl2_table.tailcut_level is None:
            raise ToolConfigurationError(f"No tailcuts level found in {self.target_dl2_file} provenance")

        data_check_files = find_datacheck_files(self.data_check_path)
        self.data_check_tables = DataCheckTables.from_files([str(path) for path in data_check_files])
        self.matching_off_runs, self.selection_bounds = select_matching_off_runs(
            self.data_check_tables.statistics,
            self.run_number,
            nsb_relative_tolerance=self.nsb_relative_tolerance,
            zenith_tolerance_deg=self.zenith_tolerance_deg,
            month_day_tolerance=self.month_day_tolerance,
        )
        if self.matching_off_runs.empty:
            self.log.warning("No off runs matched the NSB, zenith, and MMDD selection bounds; offdl2 will be empty")
        self.offrun_dl1_files = find_compatible_offrun_dl1_files(
            self.offrun_dl1_path,
            self.matching_off_runs["run_number"].astype(int).tolist(),
            self.target_dl2_table.tailcut_level,
        )
        if not self.offrun_dl1_files and not self.matching_off_runs.empty:
            self.log.warning(
                "Off runs matched the DataCheck selection, but no DL1 files passed discovery and tailcut checks; "
                "offdl2 will be empty"
            )
        self.irf_nodes = find_dl2_mc_path(
            self.mc_dl2_path,
            self.target_dl2_table,
            gh_efficiency=self.gh_efficiency,
        )
        if not self.irf_nodes:
            self.log.warning("No matching MC DL2 files were found; no IRFs will be linked")

        self.workrun_dir = Path(self.output_dir).expanduser().resolve() / f"Run{self.run_number:05d}"
        self.offrun_dl2_dir = self.workrun_dir / "offdl2"

    def start(self) -> None:
        self.workrun_dir.mkdir(parents=True, exist_ok=True)
        target_link = self.workrun_dir / self.target_dl2_file.name
        create_safe_link(self.target_dl2_file, target_link)

        generated_offrun_dl2_files = reconstruct_off_runs(
            self.offrun_dl1_files,
            Path(self.target_dl2_table.model_directory).expanduser().resolve(),
            self.offrun_dl2_dir,
            parent=self,
        )
        if self.offrun_dl1_files and not generated_offrun_dl2_files:
            self.log.warning("No off-run DL2 files were generated; inspect the DL1ToDL2Tool messages")

        irf_generator = IRFGenerator(str(Path(self.irf_output_dir).expanduser().resolve()))
        linked_irf_nodes = []
        for node in self.irf_nodes:
            irf_file = irf_generator.make_irf_node(node)
            irf_node_link = self.workrun_dir / "irf" / node.pointing_name
            create_safe_link(irf_file.parent, irf_node_link)
            linked_irf_nodes.append(irf_node_link)

        config_path = self.workrun_dir / "workrun.toml"
        write_workrun_config(
            config_path,
            {
                "target": {
                    "run_number": self.run_number,
                    "dl2_file": self.target_dl2_file,
                    "dl2_link": target_link,
                    "model_directory": self.target_dl2_table.model_directory,
                    "tailcuts": list(self.target_dl2_table.tailcut_level or ()),
                    "nsb_level": self.target_dl2_table.nsb_level,
                    "declination_line": self.target_dl2_table.dec_line,
                    "intensity_cut": float(self.target_dl2_table.intensity_cuts),
                },
                "offrun_selection": {
                    "nsb_relative_tolerance": self.nsb_relative_tolerance,
                    "nsb_min": self.selection_bounds.nsb_min,
                    "nsb_max": self.selection_bounds.nsb_max,
                    "zenith_tolerance_deg": self.zenith_tolerance_deg,
                    "zenith_min_deg": self.selection_bounds.zenith_min_deg,
                    "zenith_max_deg": self.selection_bounds.zenith_max_deg,
                    "month_day_tolerance": self.month_day_tolerance,
                    "month_day_min": self.selection_bounds.month_day_min,
                    "month_day_max": self.selection_bounds.month_day_max,
                    "matching_run_numbers": self.matching_off_runs["run_number"].astype(int).tolist(),
                    "compatible_dl1_files": self.offrun_dl1_files,
                    "generated_dl2_files": generated_offrun_dl2_files,
                },
                "irf": {
                    "mc_dl2_path": Path(self.mc_dl2_path).expanduser().resolve(),
                    "output_dir": Path(self.irf_output_dir).expanduser().resolve(),
                    "gh_efficiency": self.gh_efficiency,
                    "node_count": len(self.irf_nodes),
                    "linked_nodes": linked_irf_nodes,
                },
            },
        )

        self.log.info("Work-run directory: %s", self.workrun_dir)
        self.log.info("Compatible off-run DL1 files: %d", len(self.offrun_dl1_files))
        self.log.info("Linked IRF nodes: %d", len(linked_irf_nodes))


def main() -> None:
    BuildWorkRun().run()


if __name__ == "__main__":
    main()
