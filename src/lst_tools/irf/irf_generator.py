import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

from .config import IRFConfig
from .layout import irf_file_path
from .samples import GammaMCSample


class IRFGenerator:
    """Generate IRF files from gamma MC samples under one ``IRFConfig``."""

    def __init__(self, base_path: str | Path, irf_config: IRFConfig):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.irf_config = irf_config

    def irf_file(self, sample: GammaMCSample) -> Path:
        return self.base_path / irf_file_path(
            sample,
            irf_type=self.irf_config.irf_type,
            cut_label=self.irf_config.cut_label,
        )

    def _config_file(self, sample: GammaMCSample) -> Path:
        return self.irf_file(sample).parent / f"{sample.pointing_name}.config.json"

    def _log_file(self, sample: GammaMCSample) -> Path:
        return self.irf_file(sample).parent / f"{sample.pointing_name}.log"

    def _validate_sample(self, sample: GammaMCSample) -> None:
        if self.irf_config.point_like and sample.mc_type == "diffuse":
            raise ValueError("Point-like IRFs require a point-source gamma MC sample; got a diffuse sample")

    def _can_reuse_irf(self, sample: GammaMCSample, config: dict) -> bool:
        """True when the IRF exists and was generated with the same config."""
        if not self.irf_file(sample).is_file():
            return False
        config_file = self._config_file(sample)
        if not config_file.is_file():
            return False
        with open(config_file) as file_handle:
            return json.load(file_handle) == config

    def generate_all(self, samples: Iterable[GammaMCSample]) -> list[Path]:
        """Create or reuse IRFs for all samples with this generator's configuration."""
        return [self.generate(sample) for sample in samples]

    def generate(self, sample: GammaMCSample) -> Path:
        """Create or reuse the IRF for one sample and return its FITS file path."""
        self._validate_sample(sample)
        irf_file = self.irf_file(sample)
        config = self.irf_config.to_irf_config()
        if self._can_reuse_irf(sample, config):
            return irf_file.resolve()

        irf_file.parent.mkdir(parents=True, exist_ok=True)
        config_file = self._config_file(sample)
        with open(config_file, "w") as file_handle:
            json.dump(config, file_handle, indent=4)

        subprocess.run(
            [
                "lstchain_create_irf_files",
                f"--config={config_file}",
                f"--input-gamma-dl2={sample.path}",
                f"--output-irf-file={irf_file}",
                f"--log-file={self._log_file(sample)}",
                "--overwrite",
            ],
            check=True,
        )
        if not irf_file.is_file():
            raise RuntimeError(f"IRF command completed without creating {irf_file}")
        return irf_file.resolve()
