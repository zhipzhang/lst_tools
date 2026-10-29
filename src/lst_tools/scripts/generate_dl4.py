"""Generate one MapDataset or SpectrumDatasetOnOff per DL3 observation."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from lst_tools.dl4.config import DL4DatasetConfig
from lst_tools.dl4.dataset_producer import DL4DatasetProducer


def load_config(path: Path) -> dict:
    with path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="TOML configuration file")
    parser.add_argument("-c", "--config", dest="config_option", type=Path, help="TOML configuration file")
    parser.add_argument(
        "-i",
        "--input-directory",
        type=Path,
        help="DL3 input directory (overrides data.directory from the config)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        help="DL4 output directory (overrides output.directory from the config)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_file = args.config_option or args.config_file
    if config_file is None:
        parser.error("a configuration file is required")

    config = DL4DatasetConfig.parse(load_config(config_file))
    if args.input_directory is not None:
        config = replace(config, data=replace(config.data, directory=args.input_directory.expanduser()))
    if args.output_directory is not None:
        config = replace(config, output=replace(config.output, directory=args.output_directory.expanduser()))
    summary = DL4DatasetProducer(config).produce()
    for result in summary.results:
        message = f"{result.obs_id}: {result.status}"
        if result.output is not None:
            message += f" {result.output}"
        if result.error is not None:
            message += f" ({result.error})"
        print(message)
    for item, count in sorted(summary.counts.items()):
        print(f"{item}: {count}")


if __name__ == "__main__":
    main()
