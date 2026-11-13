"""Reduce DL2 runs to DL3 observations via lstchain's DataReductionFITSWriter."""

import warnings
from collections.abc import Iterable
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.paths import dl2_to_dl3_filename
from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
from traitlets.config import Config

from lst_tools.dl2 import LSTObservedDL2Table
from lst_tools.irf import IRFConfig, irf_grid_path

__all__ = ["DL3Reducer"]

# The interpolation grid of one MC group: all pointings of one
# dec_<y>/nsb_<x>/<irf_type>/<cut_label>/ directory, as written by
# IRFGenerator.
IRF_FILE_PATTERN = "theta_*_az_*.irf.fits.gz"


class DL3Reducer:
    """Reduces DL2 runs to DL3 observations.

    Every DL2 file is reduced against the IRF interpolation grid of its own
    MC group (NSB tuning and declination line, read from the DL2
    provenance): ``irfs_root / dec_<line> / nsb_<nsb> / <irf_type> /
    <cut_label>``. The gammaness cut rides inside the IRF; the quality
    filters come from the same ``IRFConfig`` that generated it.

    The reducer knows nothing about zenith binning: the caller chooses the
    output directory.
    """

    def __init__(
        self,
        irfs_root: str | Path,
        output_dir: str | Path,
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
        self.output_dir = Path(output_dir)
        self.source_name = source_name
        self.source_ra = float(source_ra)
        self.source_dec = float(source_dec)
        self.irf_config = irf_config
        self.interp_method = interp_method
        self.use_nearest_irf_node = use_nearest_irf_node
        self.overwrite = overwrite

    def reduce(self, dl2_files: Iterable[str | Path]) -> list[Path]:
        """Reduce a set of DL2 files into ``self.output_dir``.

        A file whose reduction fails is skipped with a warning, so one bad
        run does not abort the rest; the returned list holds the DL3 files
        that exist afterwards.
        """
        reduced = []
        for dl2_file in dl2_files:
            try:
                reduced.append(self.reduce_run(dl2_file))
            except Exception as error:  # noqa: BLE001 - one bad run must not abort the batch
                warnings.warn(f"DL3 reduction failed for {dl2_file}: {error}", stacklevel=2)
        return reduced

    def reduce_run(self, dl2_file: str | Path) -> Path:
        """One DL2 file -> one DL3 observation in ``self.output_dir``."""
        dl2_file = Path(dl2_file)
        output_file = self.output_dir / dl2_to_dl3_filename(dl2_file)
        if output_file.exists() and not self.overwrite:
            return output_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_tool(
            DataReductionFITSWriter(
                config=self._writer_config(),
                input_dl2=dl2_file,
                output_dl3_path=self.output_dir,
                input_irf_path=self._irf_grid_dir(dl2_file),
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
        return Config(self.irf_config.to_dl3_config())

    def _irf_grid_dir(self, dl2_file: Path) -> Path:
        """The IRF interpolation grid of the MC group matching the DL2 provenance.

        Uses the same layout as ``IRFGenerator``, so the reduction reads
        exactly the IRFs generated for its own selection.
        """
        table = LSTObservedDL2Table(dl2_file)
        if table.nsb_level is None or table.dec_line is None:
            raise ValueError(f"Could not derive the MC group (NSB tuning, declination line) from {dl2_file}")
        irf_dir = self.irfs_root / irf_grid_path(
            nsb_level=table.nsb_level,
            dec_line=table.dec_line,
            irf_type=self.irf_config.irf_type,
            cut_label=self.irf_config.cut_label,
        )
        if not irf_dir.is_dir():
            raise FileNotFoundError(f"No IRF grid for {dl2_file}: {irf_dir} does not exist")
        return irf_dir
