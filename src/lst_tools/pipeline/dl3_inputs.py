"""Input discovery for DL3 reduction."""

import warnings
from os import PathLike
from pathlib import Path

from lst_tools.finder import parse_run_id

from .selection import RunSelection


def find_selected_dl2_files(
    selection: RunSelection,
    dl2_directory: str | PathLike[str],
) -> list[Path]:
    """Find selected DL2 files to reduce.

    Selected runs without exactly one DL2 file are skipped with a warning.
    """
    selected_runs = list(selection)
    if not selected_runs:
        return []

    dl2_directory = Path(dl2_directory)
    if not dl2_directory.is_dir():
        run_numbers = ", ".join(str(run.run_number) for run in selected_runs)
        warnings.warn(
            f"dl3: no DL2 directory at {dl2_directory}; skipping selected runs: {run_numbers}",
            stacklevel=2,
        )
        return []

    files_by_run: dict[int, list[Path]] = {}
    for path in sorted(dl2_directory.glob("dl2_*.h5")):
        if not path.is_file():
            continue
        try:
            run_number = parse_run_id(path.name)
        except ValueError:
            continue
        files_by_run.setdefault(run_number, []).append(path)

    dl2_files = []
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

        dl2_files.append(matches[0].absolute())

    skipped = []
    if missing_run_numbers:
        skipped.append("missing DL2: " + ", ".join(map(str, missing_run_numbers)))
    if ambiguous_run_numbers:
        skipped.append("ambiguous DL2: " + ", ".join(map(str, ambiguous_run_numbers)))
    if skipped:
        warnings.warn("dl3: skipping selected runs\n" + "\n".join(skipped), stacklevel=2)

    return dl2_files
