import json
import subprocess
from pathlib import Path

from .config import IRFConfig
from .irf_nodes import IRFNode


class IRFGenerator:
    """Generate IRF files for IRF nodes under one ``IRFConfig``.

    Each node maps to one IRF file at ``base_path / node.relative_path``;
    the config and log of the generating ``lstchain_create_irf_files``
    command are written next to it as ``<pointing_name>.config.json`` /
    ``<pointing_name>.log``.
    """

    def __init__(self, base_path: str | Path, irf_config: IRFConfig):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.irf_config = irf_config

    def irf_file(self, node: IRFNode) -> Path:
        return self.base_path / node.relative_path

    def _config_file(self, node: IRFNode) -> Path:
        return self.irf_file(node).parent / f"{node.pointing_name}.config.json"

    def _log_file(self, node: IRFNode) -> Path:
        return self.irf_file(node).parent / f"{node.pointing_name}.log"

    def _is_fresh(self, node: IRFNode, config: dict) -> bool:
        """True when the IRF exists and was generated with the same config."""
        if not self.irf_file(node).is_file():
            return False
        config_file = self._config_file(node)
        if not config_file.is_file():
            return False
        with open(config_file) as file_handle:
            return json.load(file_handle) == config

    def make_irf_node(self, node: IRFNode) -> Path:
        """Create the node's IRF if needed and return its FITS file path.

        An existing IRF is reused only when it was generated with the same
        config (event selection, binning, gammaness mode); otherwise it is
        regenerated.
        """
        irf_file = self.irf_file(node)
        config = self.irf_config.to_lstchain_config()
        if self._is_fresh(node, config):
            return irf_file.resolve()

        irf_file.parent.mkdir(parents=True, exist_ok=True)
        config_file = self._config_file(node)
        with open(config_file, "w") as file_handle:
            json.dump(config, file_handle, indent=4)

        subprocess.run(
            [
                "lstchain_create_irf_files",
                f"--config={config_file}",
                f"--input-gamma-dl2={node.dl2_path}",
                f"--output-irf-file={irf_file}",
                f"--log-file={self._log_file(node)}",
                "--overwrite",
            ],
            check=True,
        )
        if not irf_file.is_file():
            raise RuntimeError(f"IRF command completed without creating {irf_file}")
        return irf_file.resolve()
