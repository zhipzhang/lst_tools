import json
import subprocess
import warnings
from pathlib import Path

from .irf_nodes import IRFNode

default_config_path = Path(__file__).parent / "default_config.json"


class IRFGenerator:
    def __init__(self, base_path: str, name_style="irf.fits.gz"):
        if Path(base_path).exists():
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)
        self.name_style = name_style

    def _irf_node_exist(self, node: IRFNode):
        path_exist = (self.base_path / node.path_name).exists()
        file_exist = (self.base_path / node.path_name / self.name_style).exists()
        if not path_exist:
            return False
        if not file_exist:
            warnings.warn(f"IRF file not found: {self.base_path / node.path_name / self.name_style}", UserWarning)
            return False
        return True

    def make_irf_node(self, node: IRFNode) -> Path:
        """Create an IRF node if needed and return its FITS file path."""
        if not default_config_path.exists():
            raise FileNotFoundError(f"Default config file not found: {default_config_path}")
        if node.dl2_path is None:
            raise ValueError("IRF node does not have an input DL2 path")
        if node.gh_efficiency is None:
            raise ValueError("IRF node does not have a gamma efficiency")

        node_path = self.base_path / node.path_name
        irf_file = node_path / self.name_style
        if irf_file.exists():
            return irf_file.resolve()

        node_path.mkdir(parents=True, exist_ok=True)
        with open(default_config_path) as f:
            self.config = json.load(f)

        # Update the lower intensity cut while preserving the upper bound.
        self.config["EventSelector"]["filters"]["intensity"][0] = node.intensity_cuts

        # Update the gh_efficiency in DL3cuts
        self.config["DL3Cuts"]["gh_efficiency"] = node.gh_efficiency
        config_file_path = node_path / "config.json"
        log_file = node_path / "log.txt"
        with open(config_file_path, "w") as f:
            json.dump(self.config, f, indent=4)

        # Running the `lstchain_create_irf_files` command
        subprocess.run(
            [
                "lstchain_create_irf_files",
                "--config=" + str(config_file_path),
                "--input-gamma-dl2=" + str(node.dl2_path),
                "--log-file=" + str(log_file),
                "--gh-efficiency=" + str(node.gh_efficiency),
                "--output-irf-file=" + str(irf_file),
                "--energy-dependent-gh",
            ],
            check=True,
        )
        if not irf_file.is_file():
            raise RuntimeError(f"IRF command completed without creating {irf_file}")
        return irf_file.resolve()
