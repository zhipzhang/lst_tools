"""Reconstruct DL2 files from DL1 inputs using lstchain Random Forest models."""

from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.tools.lstchain_dl1_to_dl2 import DL1ToDL2Tool

from lst_tools.dl1.nsb import resolve_dl1_nsb_level


# REFACTOR: $NSB_LEVEL also has a constant in pipeline.configs. Use one vocabulary and
# test this planning function directly: literal template, resolved groups, and input order.
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


# REFACTOR: Keep DL1ToDL2Tool/run_tool behind this single external boundary. Make produced
# output paths explicit to callers and tests.
def dl1_to_dl2_batch(files: Iterable[str | PathLike[str]], out_root, model_directory, overwrite=False):
    input_files = [Path(file).resolve() for file in files]
    for input_file in input_files:
        if not input_file.is_file():
            raise FileNotFoundError(f"File not found: {input_file}")
    tool = DL1ToDL2Tool(
        input_files=input_files,
        path_models=Path(model_directory),
        output_dir=Path(out_root),
        overwrite=overwrite,
    )
    run_tool(tool)


# REFACTOR: Define empty-input behavior. A literal template currently runs one empty batch,
# while an NSB template runs none; both create out_root. Also centralize DL1->DL2 output
# naming instead of repeating replace("dl1", "dl2", 1) here and in DL2Producer.
def reconstruct_dl1_to_dl2(
    files: Iterable[str | PathLike[str]],
    out_root: str | PathLike[str],
    model_directory_template: str | PathLike[str],
    overwrite: bool = False,
) -> None:
    """Reconstruct DL1 files, grouping them by their resolved RF model directory."""
    dl1_files = [Path(file) for file in files]
    missing_files = [file for file in dl1_files if not file.is_file()]
    if missing_files:
        raise FileNotFoundError(f"The following input files do not exist: {missing_files}")

    model_groups = group_dl1_by_model_directory(dl1_files, model_directory_template)
    missing_models = [directory for directory in model_groups if not directory.is_dir()]
    if missing_models:
        raise FileNotFoundError(f"Model directories do not exist or are not directories: {missing_models}")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for model_directory, group in model_groups.items():
        dl1_to_dl2_batch(group, out_root, model_directory, overwrite=overwrite)
