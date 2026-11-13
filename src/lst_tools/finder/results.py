from abc import ABC

from .models import DL1Metadata, DL2Metadata, DL3Metadata
from .parsing.observed import parse_dl1_path, parse_dl2_path, parse_dl3_path


def _version_key(metadata: DL1Metadata) -> tuple[int, ...]:
    return tuple(int(part) for part in metadata.lstchain_version.split("."))


class DataSearchResult(ABC):
    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]


class DL1SearchResult(DataSearchResult):
    def __init__(self, files: list[str]):
        super().__init__(files)
        self.metadata = []
        for file in self.files:
            try:
                self.metadata.append(parse_dl1_path(file))
            except ValueError as error:
                print(f"Error parsing DL1 file {file}: {error}")

    @property
    def newest_metadata(self) -> DL1Metadata | None:
        """Metadata of the file with the newest lstchain version, or None if empty."""
        if not self.metadata:
            return None
        return max(self.metadata, key=_version_key)

    @property
    def dl1_files(self):
        return self.metadata


class DL2SearchResult(DataSearchResult):
    def __init__(self, files: list[str]):
        super().__init__(files)
        self.metadata = []
        for file in self.files:
            try:
                self.metadata.append(parse_dl2_path(file))
            except ValueError as error:
                print(f"Error parsing DL2 file {file}: {error}")

    @property
    def newest_metadata(self) -> DL2Metadata | None:
        """Metadata of the file with the newest lstchain version, or None if empty."""
        if not self.metadata:
            return None
        return max(self.metadata, key=_version_key)

    @property
    def dl2_files(self):
        return self.metadata


class DL3SearchResult(DataSearchResult):
    def __init__(self, files: list[str]):
        super().__init__(files)
        self.metadata = [parse_dl3_path(file) for file in self.files]

    @property
    def newest_metadata(self) -> list[DL3Metadata]:
        """Metadata for all DL3 variants produced with the newest lstchain version.

        DL3 files come in several variants (irf type, gh/theta cuts) per version,
        so the result is a list rather than a single file. Empty if the search
        found nothing.
        """
        if not self.metadata:
            return []
        newest_version = max(_version_key(metadata) for metadata in self.metadata)
        return [metadata for metadata in self.metadata if _version_key(metadata) == newest_version]

    @property
    def dl3_files(self):
        return self.metadata


DataLevelSearchResult = DataSearchResult
DL1LevelSearchResult = DL1SearchResult
DL2LevelSearchResult = DL2SearchResult
DL3LevelSearchResult = DL3SearchResult
