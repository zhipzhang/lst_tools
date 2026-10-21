from abc import ABC

from .datalevel_attributes import DL1Attributes, DL2Attributes, DL3Attributes
from .helper import parse_dl1_path, parse_dl2_path, parse_dl3_path


def _version_key(attrs: DL1Attributes) -> tuple[int, ...]:
    return tuple(int(part) for part in attrs.lstchain_version.split("."))


class DataLevelSearchResult(ABC):
    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]


class DL1LevelSearchResult(DataLevelSearchResult):
    def __init__(self, files: list[str]):
        super().__init__(files)
        self.dl1_files = [parse_dl1_path(f) for f in self.files]

    @property
    def newest_attributes(self) -> DL1Attributes | None:
        """Attributes of the file with the newest lstchain version, or None if empty."""
        if not self.dl1_files:
            return None
        return max(self.dl1_files, key=_version_key)


class DL2LevelSearchResult(DataLevelSearchResult):
    def __init__(self, files: list[str]):
        super().__init__(files)
        self.dl2_files = [parse_dl2_path(f) for f in self.files]

    @property
    def newest_attributes(self) -> DL2Attributes | None:
        """Attributes of the file with the newest lstchain version, or None if empty."""
        if not self.dl2_files:
            return None
        return max(self.dl2_files, key=_version_key)


class DL3LevelSearchResult(DataLevelSearchResult):
    def __init__(self, files: list[str]):
        super().__init__(files)
        self.dl3_files = [parse_dl3_path(f) for f in self.files]

    @property
    def newest_attributes(self) -> list[DL3Attributes]:
        """All DL3 variants produced with the newest lstchain version.

        DL3 files come in several variants (irf type, gh/theta cuts) per version,
        so the result is a list rather than a single file. Empty if the search
        found nothing.
        """
        if not self.dl3_files:
            return []
        newest_version = max(_version_key(attrs) for attrs in self.dl3_files)
        return [attrs for attrs in self.dl3_files if _version_key(attrs) == newest_version]
