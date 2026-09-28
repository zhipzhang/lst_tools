import re

from .dl2table_base import LSTDL2TableBase


class LSTDL2MCTable(LSTDL2TableBase):
    """DL2 MC events loaded from one lstchain MC file."""

    @property
    def pointing_ze(self) -> float:
        match = re.search(r"node_theta_([\d.]+)", self.file_name)
        if match is None:
            raise ValueError(f"Could not parse pointing zenith from MC path: {self.file_name}")
        return float(match.group(1))

    @property
    def pointing_az(self) -> float:
        match = re.search(r"node_theta_[\d.]+_az_([\d.]+)", self.file_name)
        if match is None:
            raise ValueError(f"Could not parse pointing azimuth from MC path: {self.file_name}")
        return float(match.group(1))
