"""Build a radial DL2 background model from reconstructed off runs.

The target DL2 supplies the intensity cut, the target DL3 supplies the
energy-dependent gammaness cuts, and the selected off-run DL2 files supply
the events and livetime used to construct the background rate.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import astropy.units as u
import numpy as np
from astropy.table import QTable
from ctapipe.core import Tool, ToolConfigurationError, traits

from lst_tools.bkg import Background2DMaker, SkyOffsetEvents, sky_offset_events_from_lstdl2
from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.event_filter import EventFilter

OFFRUN_DL2_PATTERN = "dl2_LST-1.Run*.h5"
_RUN_PATTERN = re.compile(r"Run(?P<run_number>\d+)")


def find_offrun_dl2_files(
    path: str | Path,
    *,
    exclude_run_numbers: Iterable[int] = (),
) -> list[Path]:
    """Find off-run DL2 files, excluding any configured validation runs."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise ToolConfigurationError(f"Off-run DL2 directory does not exist: {directory}")

    excluded = {int(run_number) for run_number in exclude_run_numbers}
    files = []
    for file_path in sorted(directory.glob(OFFRUN_DL2_PATTERN)):
        match = _RUN_PATTERN.search(file_path.name)
        if file_path.is_file() and match is not None and int(match.group("run_number")) not in excluded:
            files.append(file_path.resolve())

    if not files:
        exclusion = f" after excluding runs {sorted(excluded)}" if excluded else ""
        raise ToolConfigurationError(
            f"No off-run DL2 files matching {OFFRUN_DL2_PATTERN!r} found under {directory}{exclusion}"
        )
    return files


def read_gh_cuts(target_dl3_file: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read energy-dependent gammaness cuts from a target DL3 file."""
    try:
        table = QTable.read(target_dl3_file, hdu="GH_CUTS")
    except Exception as error:
        raise ToolConfigurationError(f"Could not read GH_CUTS from {target_dl3_file}: {error}") from error

    missing = {"low", "high", "cut"}.difference(table.colnames)
    if missing:
        raise ToolConfigurationError(f"GH_CUTS is missing columns: {sorted(missing)}")

    def energy_values(column_name: str) -> np.ndarray:
        column = table[column_name]
        if getattr(column, "unit", None) is not None:
            return np.asarray(u.Quantity(column).to_value(u.TeV), dtype=float)
        return np.asarray(column, dtype=float)

    return (
        energy_values("low"),
        energy_values("high"),
        np.asarray(table["cut"], dtype=float),
    )


def build_dl2_background(
    target_dl2_file: str | Path,
    target_dl3_file: str | Path,
    offrun_dl2_path: str | Path,
    output_file: str | Path,
    *,
    energy_edges: Iterable[float],
    theta_edges: Iterable[float],
    spectral_index: float = -2.0,
    exclude_run_numbers: Iterable[int] = (),
    overwrite: bool = False,
) -> Path:
    """Filter off-run events and write their binned radial background model."""
    target_dl2_path = Path(target_dl2_file).expanduser().resolve()
    target_dl3_path = Path(target_dl3_file).expanduser().resolve()
    if not target_dl2_path.is_file():
        raise ToolConfigurationError(f"Target DL2 file does not exist: {target_dl2_path}")
    if not target_dl3_path.is_file():
        raise ToolConfigurationError(f"Target DL3 file does not exist: {target_dl3_path}")

    background_maker = Background2DMaker(
        energy_edges=energy_edges,
        theta_edges=theta_edges,
        spectral_index=spectral_index,
    )
    offrun_files = find_offrun_dl2_files(offrun_dl2_path, exclude_run_numbers=exclude_run_numbers)
    output_path = Path(output_file).expanduser().resolve()
    if output_path.is_file() and not overwrite:
        return output_path

    target_dl2 = LSTDL2EventTable(target_dl2_path)
    energy_low, energy_high, gh_cuts = read_gh_cuts(target_dl3_path)
    event_filter = EventFilter(
        intensity_cuts=target_dl2.intensity_cuts,
        energy_low=energy_low,
        energy_high=energy_high,
        gh_cuts=gh_cuts,
    )

    events = SkyOffsetEvents()
    for offrun_file in offrun_files:
        offrun_dl2 = LSTDL2EventTable(offrun_file)
        events = events + sky_offset_events_from_lstdl2(offrun_dl2, event_filter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background_maker(events).write(output_path, overwrite=overwrite)
    if not output_path.is_file():
        raise RuntimeError(f"Background2DMaker did not create {output_path}")
    return output_path


class BuildDL2Background(Tool):
    """Command-line interface for building one radial DL2 background model."""

    name = "build-dl2-background"
    description = __doc__

    target_dl2_file = traits.Path(help="Target DL2 file that supplies the intensity cut").tag(config=True)
    target_dl3_file = traits.Path(help="Target DL3 file containing the GH_CUTS table").tag(config=True)
    offrun_dl2_path = traits.Path(help="Directory containing reconstructed off-run DL2 files").tag(config=True)
    output_file = traits.Path(default_value="./background2d.fits", help="Output Background2D FITS file").tag(
        config=True
    )
    energy_edges = traits.List(
        trait=traits.Float(),
        default_value=[],
        help="Reconstructed-energy bin edges in TeV",
    ).tag(config=True)
    theta_edges = traits.List(
        trait=traits.Float(),
        default_value=[],
        help="Radial offset bin edges in degrees",
    ).tag(config=True)
    spectral_index = traits.Float(
        default_value=-2.0,
        help="Spectral index used for differential-rate normalization",
    ).tag(config=True)
    exclude_run_numbers = traits.List(
        trait=traits.Int(),
        default_value=[],
        help="Off-run numbers reserved for validation and excluded from training",
    ).tag(config=True)
    overwrite = traits.Bool(default_value=False, help="Overwrite an existing background file").tag(config=True)

    aliases: ClassVar[dict[tuple[str, ...], str]] = {
        ("target-dl2",): "BuildDL2Background.target_dl2_file",
        ("target-dl3",): "BuildDL2Background.target_dl3_file",
        ("offrun-dl2",): "BuildDL2Background.offrun_dl2_path",
        ("o", "output"): "BuildDL2Background.output_file",
        ("energy-edges",): "BuildDL2Background.energy_edges",
        ("theta-edges",): "BuildDL2Background.theta_edges",
        ("spectral-index",): "BuildDL2Background.spectral_index",
        ("exclude-run",): "BuildDL2Background.exclude_run_numbers",
    }
    flags: ClassVar[dict[str, tuple[dict[str, dict[str, bool]], str]]] = {
        "overwrite": (
            {"BuildDL2Background": {"overwrite": True}},
            "Overwrite an existing background file",
        ),
    }

    def setup(self) -> None:
        if not self.energy_edges:
            raise ToolConfigurationError("energy_edges must be configured")
        if not self.theta_edges:
            raise ToolConfigurationError("theta_edges must be configured")
        try:
            Background2DMaker(
                energy_edges=self.energy_edges,
                theta_edges=self.theta_edges,
                spectral_index=self.spectral_index,
            )
        except (TypeError, ValueError) as error:
            raise ToolConfigurationError(f"Invalid background binning: {error}") from error

    def start(self) -> None:
        self.background_file = build_dl2_background(
            self.target_dl2_file,
            self.target_dl3_file,
            self.offrun_dl2_path,
            self.output_file,
            energy_edges=self.energy_edges,
            theta_edges=self.theta_edges,
            spectral_index=self.spectral_index,
            exclude_run_numbers=self.exclude_run_numbers,
            overwrite=self.overwrite,
        )
        self.log.info("DL2 background file: %s", self.background_file)


def main() -> None:
    BuildDL2Background().run()


if __name__ == "__main__":
    main()
