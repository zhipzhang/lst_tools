"""Build IRF grids and reduce DL2 input files to DL3 observations.

Intended as both a standalone CLI tool and a worker entrypoint for SLURM jobs.
"""

import argparse
from collections import Counter
from pathlib import Path

from lst_tools.irf import IRFConfig
from lst_tools.pipeline.dl3_build import DL3Build


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
        help="One or more DL2 input .h5 files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Output directory where DL3 files will be written",
    )
    parser.add_argument(
        "--irf-dir",
        dest="irf_dir",
        type=Path,
        required=True,
        help="Root directory where IRF interpolation grids are stored",
    )
    parser.add_argument(
        "--irf-config",
        dest="irf_config",
        type=Path,
        required=True,
        help="JSON configuration file for IRF generation and DL3 cuts",
    )
    parser.add_argument(
        "--mc-dir",
        dest="mc_dir",
        type=str,
        required=True,
        help="MC DL2 campaign directory or template containing $NSB_LEVEL",
    )
    parser.add_argument(
        "--source-name",
        dest="source_name",
        type=str,
        required=True,
        help="Analysis target source name",
    )
    parser.add_argument(
        "--source-ra",
        dest="source_ra",
        type=float,
        required=True,
        help="Target source right ascension in degrees",
    )
    parser.add_argument(
        "--source-dec",
        dest="source_dec",
        type=float,
        required=True,
        help="Target source declination in degrees",
    )
    parser.add_argument(
        "--interp-method",
        dest="interp_method",
        type=str,
        default="linear",
        help="IRF interpolation method",
    )
    parser.add_argument(
        "--use-nearest-irf-node",
        action="store_true",
        default=False,
        help="Use the nearest IRF node instead of interpolating",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing DL3 files in the output directory",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker processes",
    )
    parser.add_argument(
        "--runs-per-worker",
        type=int,
        default=5,
        help="Number of DL2 runs reduced per worker batch",
    )
    return parser


def main(argv: list[str] | None = None) -> Counter:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.runs_per_worker <= 0:
        parser.error("--runs-per-worker must be positive")
    if not args.irf_config.is_file():
        parser.error(f"IRF config file not found: {args.irf_config}")

    irf_config = IRFConfig.from_json(args.irf_config)
    build = DL3Build(
        dl2_files=args.input_files,
        output_dir=args.output_dir,
        irfs_root=args.irf_dir,
        irf_config=irf_config,
        mc_dl2_directory=args.mc_dir,
        source_name=args.source_name,
        source_ra=args.source_ra,
        source_dec=args.source_dec,
        interp_method=args.interp_method,
        use_nearest_irf_node=args.use_nearest_irf_node,
        overwrite=args.overwrite,
    )
    counts = build.run(
        max_workers=args.max_workers,
        runs_per_worker=args.runs_per_worker,
    )
    for item, count in sorted(counts.items()):
        print(f"  {item}: {count}")
    return counts


if __name__ == "__main__":
    main()
