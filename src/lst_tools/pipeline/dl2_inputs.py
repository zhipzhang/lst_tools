"""Input discovery for DL2 reconstruction."""

import warnings
from os import PathLike
from pathlib import Path

from lst_tools.dl2.reconstruction import dl2_output_path
from lst_tools.finder import parse_run_id

from .selection import RunSelection


def find_pending_dl1_files(
    selection: RunSelection,
    dl1_directory: str | PathLike[str],
    output_directory: str | PathLike[str],
    overwrite: bool = False,
) -> tuple[list[Path], int]:
    """Find selected DL1 files to reconstruct and count reusable DL2 outputs.

    Selected runs without exactly one DL1 file are skipped with a warning.
    Existing DL2 outputs are skipped unless ``overwrite`` is true.
    """
    selected_runs = list(selection)
    if not selected_runs:
        return [], 0

    dl1_directory = Path(dl1_directory)
    if not dl1_directory.is_dir():
        run_numbers = ", ".join(str(run.run_number) for run in selected_runs)
        warnings.warn(
            f"dl2: no DL1 directory at {dl1_directory}; skipping selected runs: {run_numbers}",
            stacklevel=2,
        )
        return [], 0

    files_by_run: dict[int, list[Path]] = {}
    for path in sorted(dl1_directory.glob("dl1_*.h5")):
        if not path.is_file():
            continue
        try:
            run_number = parse_run_id(path.name)
        except ValueError:
            continue
        files_by_run.setdefault(run_number, []).append(path)

    pending = []
    existing_count = 0
    missing_run_numbers = []
    ambiguous_run_numbers = []
    for run in selected_runs:
        matches = files_by_run.get(run.run_number, [])
        if not matches:
            missing_run_numbers.append(run.run_number)
            continue
        if len(matches) > 1:
            ambiguous_run_numbers.append(run.run_number)
            continue

        dl1_path = matches[0]
        if overwrite or not dl2_output_path(dl1_path, output_directory).exists():
            pending.append(dl1_path)
        else:
            existing_count += 1

    skipped = []
    if missing_run_numbers:
        skipped.append("missing DL1: " + ", ".join(map(str, missing_run_numbers)))
    if ambiguous_run_numbers:
        skipped.append("ambiguous DL1: " + ", ".join(map(str, ambiguous_run_numbers)))
    if skipped:
        warnings.warn("dl2: skipping selected runs\n" + "\n".join(skipped), stacklevel=2)

    return pending, existing_count
