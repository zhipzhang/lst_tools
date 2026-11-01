"""Reconstruct DL2 files from DL1 input files using Random Forest models.

Intended as both a standalone CLI tool and a worker entrypoint for SLURM batch jobs.
"""

import argparse
from pathlib import Path

from lst_tools.dl1.dl1event import reconstruct_dl1_to_dl2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        "--input-files",
        dest="input_files",
        nargs="+",
        type=Path,
        required=True,
        help="One or more DL1 input .h5 files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Output directory where DL2 files will be written",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        "--path-models",
        dest="model_dir",
        type=Path,
        required=True,
        help="RF model directory or template containing $NSB_LEVEL",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing DL2 files in the output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        reconstruct_dl1_to_dl2(
            files=args.input_files,
            out_root=args.output_dir,
            model_directory_template=args.model_dir,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    print(f"Reconstructed {len(args.input_files)} DL1 file(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
