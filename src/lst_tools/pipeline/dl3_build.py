"""Build IRF grids and reduce DL2 runs to DL3 observations for one reduction."""

import os
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lst_tools.dl2 import LSTObservedDL2Table
from lst_tools.dl3 import DL3Reducer
from lst_tools.finder import MC_DL2_PATH, MCDL2Metadata, find_mc_dl2_files
from lst_tools.irf import GammaMCSample, IRFConfig, IRFGenerator, select_gamma_mc_samples

from .configs import NSB_LEVEL_VARIABLE


class DL3Build:
    """Build IRF grids and reduce DL2 runs to DL3 observations for one reduction."""

    def __init__(
        self,
        dl2_files: Sequence[str | Path],
        output_dir: str | Path,
        irfs_root: str | Path,
        irf_config: IRFConfig,
        mc_dl2_directory: str,
        *,
        source_name: str,
        source_ra: float,
        source_dec: float,
        interp_method: str = "linear",
        use_nearest_irf_node: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.dl2_files = tuple(Path(path).absolute() for path in dl2_files)
        self.mc_dl2_directory = mc_dl2_directory
        self.reducer = DL3Reducer(
            irfs_root=Path(irfs_root).absolute(),
            output_dir=Path(output_dir).absolute(),
            source_name=source_name,
            source_ra=source_ra,
            source_dec=source_dec,
            irf_config=irf_config,
            interp_method=interp_method,
            use_nearest_irf_node=use_nearest_irf_node,
            overwrite=overwrite,
        )

    def run(self, max_workers: int | None = None, runs_per_worker: int = 5) -> Counter:
        if runs_per_worker <= 0:
            raise ValueError("DL3 runs_per_worker must be positive")
        workers = self._worker_limit(max_workers)
        self._validate_dl2_files()
        if not self.dl2_files:
            return Counter()

        sample_groups = self._find_sample_groups()
        self._prepare_irfs(sample_groups, workers)
        counts = self._reduce_batches(workers, runs_per_worker)
        counts["dl3:irf_files"] = sum(len(samples) for samples in sample_groups)
        return counts

    def _prepare_irfs(self, sample_groups: list[tuple[GammaMCSample, ...]], workers: int) -> None:
        irf_generator = IRFGenerator(self.reducer.irfs_root, self.reducer.irf_config)
        with ProcessPoolExecutor(max_workers=min(workers, len(sample_groups))) as pool:
            list(pool.map(irf_generator.generate_all, sample_groups))

    def _reduce_batches(self, workers: int, runs_per_worker: int) -> Counter:
        batches = [
            self.dl2_files[start : start + runs_per_worker] for start in range(0, len(self.dl2_files), runs_per_worker)
        ]
        counts: Counter = Counter()
        with ProcessPoolExecutor(max_workers=min(workers, len(batches))) as pool:
            for batch, reduced in zip(batches, pool.map(self.reducer.reduce, batches), strict=True):
                counts["dl3:reduced"] += len(reduced)
                counts["dl3:failed"] += len(batch) - len(reduced)
        return counts

    def _find_sample_groups(self) -> list[tuple[GammaMCSample, ...]]:
        mc_tunings = self._resolve_dl2_mc_tunings()
        campaigns: dict[Path, list[MCDL2Metadata]] = {}
        irf_config = self.reducer.irf_config
        irf_type = irf_config.irf_type
        finder_mc_type = "Gamma" if irf_config.point_like else "GammaDiffuse"
        sample_groups: list[tuple[GammaMCSample, ...]] = []
        for nsb_level, dec_line in mc_tunings:
            campaign = self._resolve_campaign_directory(nsb_level)
            if campaign not in campaigns:
                campaigns[campaign] = list(find_mc_dl2_files(campaign))
            samples = select_gamma_mc_samples(
                campaigns[campaign],
                nsb_level=nsb_level,
                dec_line=dec_line,
                irf_type=irf_type,
            )
            if not samples:
                raise FileNotFoundError(
                    f"No {finder_mc_type} MC samples for NSB {nsb_level:g}, declination line {dec_line} in {campaign}"
                )
            sample_groups.append(samples)
        return sample_groups

    def _resolve_dl2_mc_tunings(self) -> tuple[tuple[float, int], ...]:
        tunings = []
        for dl2_file in self.dl2_files:
            table = LSTObservedDL2Table(dl2_file)
            if table.nsb_level is None or table.dec_line is None:
                raise ValueError(f"Could not derive the MC group (NSB tuning, declination line) from {dl2_file}")
            tunings.append((table.nsb_level, table.dec_line))
        return tuple(dict.fromkeys(tunings))

    def _resolve_campaign_directory(self, nsb_level: float) -> Path:
        campaign = Path(self.mc_dl2_directory.replace(NSB_LEVEL_VARIABLE, f"{nsb_level:.2f}")).expanduser()
        if not campaign.is_absolute():
            return MC_DL2_PATH / campaign
        return campaign

    def _validate_dl2_files(self) -> None:
        missing = [str(dl2_file) for dl2_file in self.dl2_files if not dl2_file.is_file()]
        if missing:
            raise FileNotFoundError("dl3: selected DL2 files no longer exist:\n" + "\n".join(missing))

    @staticmethod
    def _worker_limit(max_workers: int | None) -> int:
        if max_workers is not None:
            if max_workers <= 0:
                raise ValueError("DL3 max_workers must be positive")
            return max_workers
        return os.cpu_count() or 1
