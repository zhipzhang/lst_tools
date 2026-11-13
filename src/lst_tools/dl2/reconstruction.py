"""Reconstruct DL2 files from DL1 inputs using lstchain Random Forest models."""

from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.tools.lstchain_dl1_to_dl2 import DL1ToDL2Tool

from lst_tools.dl1.nsb import resolve_dl1_nsb_level


def dl2_output_path(dl1_path: str | PathLike[str], output_directory: str | PathLike[str]) -> Path:
    dl1_path = Path(dl1_path)
    return Path(output_directory) / dl1_path.name.replace("dl1", "dl2", 1)


def group_dl1_by_model_directory(
    files: Iterable[str | PathLike[str]],
    model_directory_template: str | PathLike[str],
) -> dict[Path, list[Path]]:
    """Group DL1 files by RF model directory, resolving NSB per run if needed."""
    dl1_files = [Path(file) for file in files]
    template = str(model_directory_template)
    if "$NSB_LEVEL" not in template:
        return {Path(template): dl1_files}

    groups: dict[Path, list[Path]] = {}
    for dl1_file in dl1_files:
        nsb_level = resolve_dl1_nsb_level(dl1_file)
        model_directory = Path(template.replace("$NSB_LEVEL", f"{nsb_level:.2f}"))
        groups.setdefault(model_directory, []).append(dl1_file)
    return groups


def dl1_to_dl2_batch(
    files: Iterable[str | PathLike[str]],
    output_directory: str | PathLike[str],
    model_directory: str | PathLike[str],
    overwrite: bool = False,
) -> None:
    input_files = [Path(file).resolve() for file in files]
    for input_file in input_files:
        if not input_file.is_file():
            raise FileNotFoundError(f"File not found: {input_file}")
    tool = DL1ToDL2Tool(
        input_files=input_files,
        path_models=Path(model_directory),
        output_dir=Path(output_directory),
        overwrite=overwrite,
    )
    run_tool(tool)


def reconstruct_dl1_to_dl2(
    files: Iterable[str | PathLike[str]],
    output_directory: str | PathLike[str],
    model_directory_template: str | PathLike[str],
    overwrite: bool = False,
) -> list[Path]:
    """Reconstruct DL1 files and return the DL2 outputs available afterwards."""
    dl1_files = [Path(file) for file in files]
    if not dl1_files:
        return []

    missing_files = [file for file in dl1_files if not file.is_file()]
    if missing_files:
        raise FileNotFoundError(f"The following input files do not exist: {missing_files}")

    model_groups = group_dl1_by_model_directory(dl1_files, model_directory_template)
    missing_models = [directory for directory in model_groups if not directory.is_dir()]
    if missing_models:
        raise FileNotFoundError(f"Model directories do not exist or are not directories: {missing_models}")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for model_directory, group in model_groups.items():
        dl1_to_dl2_batch(group, output_directory, model_directory, overwrite=overwrite)

    output_paths = [dl2_output_path(dl1_file, output_directory) for dl1_file in dl1_files]
    return [output_path for output_path in output_paths if output_path.is_file()]
