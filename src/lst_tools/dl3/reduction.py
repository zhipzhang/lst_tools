"""Reduce DL2 runs to DL3 observations via lstchain's DataReductionFITSWriter."""

import warnings
from collections.abc import Iterable
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.paths import dl2_to_dl3_filename
from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
from traitlets.config import Config

from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.irf import IRFConfig

__all__ = ["DL3Reducer"]

# The interpolation set of one MC group: all pointings of one
# nsb_<x>/dec_<y>/ directory, as written by IRFGenerator (IRFNode.relative_path).
IRF_FILE_PATTERN = "theta_*_az_*.irf.fits.gz"


class DL3Reducer:
    """Reduces DL2 runs to DL3 observations.

    Every DL2 file is reduced against the IRF interpolation set of its own
    MC group (NSB tuning and declination line, read from the DL2
    provenance): ``irfs_root / nsb_<nsb> / dec_<line>``. The gammaness cut
    rides inside the IRF; the quality filters come from the same
    ``IRFConfig`` that generated it.

    The reducer knows nothing about zenith binning: the caller chooses the
    output directory.
    """

    def __init__(
        self,
        irfs_root: str | Path,
        source_name: str,
        source_ra: float,
        source_dec: float,
        irf_config: IRFConfig,
        *,
        interp_method: str = "linear",
        use_nearest_irf_node: bool = False,
        overwrite: bool = False,
    ):
        self.irfs_root = Path(irfs_root)
        self.source_name = source_name
        self.source_ra = float(source_ra)
        self.source_dec = float(source_dec)
        self.irf_config = irf_config
        self.interp_method = interp_method
        self.use_nearest_irf_node = use_nearest_irf_node
        self.overwrite = overwrite

    def reduce(self, dl2_files: Iterable[str | Path], output_dir: str | Path) -> list[Path]:
        """Reduce a set of DL2 files into one DL3 directory.

        A file whose reduction fails is skipped with a warning, so one bad
        run does not abort the rest; the returned list holds the DL3 files
        that exist afterwards.
        """
        output_dir = Path(output_dir)
        reduced = []
        for dl2_file in dl2_files:
            try:
                reduced.append(self.reduce_run(dl2_file, output_dir))
            except Exception as error:  # noqa: BLE001 - one bad run must not abort the batch
                warnings.warn(f"DL3 reduction failed for {dl2_file}: {error}", stacklevel=2)
        return reduced

    def reduce_run(self, dl2_file: str | Path, output_dir: str | Path) -> Path:
        """One DL2 file -> one DL3 observation."""
        dl2_file = Path(dl2_file)
        output_dir = Path(output_dir)
        output_file = output_dir / dl2_to_dl3_filename(dl2_file)
        if output_file.exists() and not self.overwrite:
            return output_file

        output_dir.mkdir(parents=True, exist_ok=True)
        run_tool(
            DataReductionFITSWriter(
                config=self._writer_config(),
                input_dl2=dl2_file,
                output_dl3_path=output_dir,
                input_irf_path=self._irf_set_dir(dl2_file),
                irf_file_pattern=IRF_FILE_PATTERN,
                source_name=self.source_name,
                source_ra=f"{self.source_ra} deg",
                source_dec=f"{self.source_dec} deg",
                interp_method=self.interp_method,
                use_nearest_irf_node=self.use_nearest_irf_node,
                overwrite=self.overwrite,
            )
        )
        if not output_file.exists():
            raise RuntimeError(f"DataReductionFITSWriter did not create {output_file}")
        return output_file

    def _writer_config(self) -> Config:
        """The EventSelector/DL3Cuts sections of the IRF config, for the writer."""
        sections = self.irf_config.to_lstchain_config()
        return Config({name: sections[name] for name in ("EventSelector", "DL3Cuts") if name in sections})

    def _irf_set_dir(self, dl2_file: Path) -> Path:
        """The IRF interpolation set of the MC group matching the DL2 provenance.

        Same naming contract as ``IRFNode.relative_path``
        (``nsb_<nsb>/dec_<line>``).
        """
        table = LSTDL2EventTable(dl2_file)
        if table.nsb_level is None or table.dec_line is None:
            raise ValueError(f"Could not derive the MC group (NSB tuning, declination line) from {dl2_file}")
        dec_name = f"dec_min_{abs(table.dec_line)}" if table.dec_line < 0 else f"dec_{table.dec_line}"
        irf_dir = self.irfs_root / f"nsb_{table.nsb_level:g}" / dec_name
        if not irf_dir.is_dir():
            raise FileNotFoundError(f"No IRF set for {dl2_file}: {irf_dir} does not exist")
        return irf_dir
