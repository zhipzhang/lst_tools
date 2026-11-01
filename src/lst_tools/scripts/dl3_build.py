"""Run selected DL3 reductions from a trusted pickled DL3Build."""

import argparse
import pickle
from collections import Counter
from pathlib import Path

from lst_tools.pipeline.dl3 import DL3Build


def build_dl3(build_pickle: str | Path, max_workers: int | None = None) -> Counter:
    """Load the selected observations and run their configured reductions."""
    with Path(build_pickle).open("rb") as file_handle:
        build = pickle.load(file_handle)
    if not isinstance(build, DL3Build):
        raise TypeError(f"Expected a pickled DL3Build, got {type(build).__name__}")
    return build.run(max_workers=max_workers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--build-pickle", type=Path, required=True)
    parser.add_argument("--max-workers", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    counts = build_dl3(args.build_pickle, args.max_workers)
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
