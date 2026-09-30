import re
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.tools.lstchain_dl1_to_dl2 import DL1ToDL2Tool

from lst_tools.dl2 import LSTDL2EventTable


class LSTDL1Event:
    def __init__(self, file_name: str | PathLike[str]):
        self.file_name = Path(file_name)
        if not self.file_name.exists():
            raise FileNotFoundError(f"File not found: {self.file_name}")

        self.run_id = self._parse_run_id(Path(file_name).name)

    @staticmethod
    def _parse_run_id(file_name: str) -> int:
        match = re.search(r"Run(?P<run_id>\d+)", file_name)
        if match is None:
            raise ValueError(f"could not find a run number in DL1 filename: {file_name}")
        return int(match.group("run_id"))

    def to_dl2(self, out_root: str | PathLike[str], model_directory: str | PathLike[str]) -> LSTDL2EventTable:
        if not Path(model_directory).exists():
            raise FileNotFoundError(f"Model directory not found: {model_directory}")
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        reconstructor = DL1ToDL2Tool(
            input_files=[self.file_name],
            path_models=Path(model_directory),
            output_dir=out_root,
        )
        run_tool(reconstructor)
        dl2_file_name = self.file_name.name.replace("dl1", "dl2")
        if not Path(out_root / dl2_file_name).exists():
            raise FileNotFoundError(f"DL2 file not found: {out_root / dl2_file_name}")
        return LSTDL2EventTable(out_root / dl2_file_name)


def dl1_to_dl2_batch(files: Iterable[str | PathLike[str]], out_root, model_directory, overwrite=False):
    events = [LSTDL1Event(f) for f in files]  # validates existence + run id up front
    tool = DL1ToDL2Tool(
        input_files=[e.file_name for e in events],
        path_models=Path(model_directory),
        output_dir=Path(out_root),
        overwrite=overwrite,
    )
    run_tool(tool)
