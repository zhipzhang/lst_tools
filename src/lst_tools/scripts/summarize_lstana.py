"""Summarize the selected runs of an init-lstana analysis as a table."""

import argparse
import warnings
from pathlib import Path

import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from lst_tools.datacheck import DataCheckTables
from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.pipeline import PipelineConfig
from lst_tools.scripts.init_lstana import load_config, make_producers, selection_file

ANGLE_COLUMNS = ("RA_PNT", "DEC_PNT", "AZ_PNT", "ZE_PNT", "B_DELTA")
COLUMNS = ("run_number", "date", *ANGLE_COLUMNS, "LIVETIME", "N_EVENTS", "NSB_LEVEL")


def _find_run_file(directory: Path, level: str, run_number: int) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.rglob(f"{level}_*Run{run_number:05d}*"))
    matches = [path for path in matches if path.is_file()]
    return matches[0] if matches else None


def _b_delta_deg(dl3_file: Path) -> float:
    try:
        header = fits.getheader(dl3_file, extname="POINTING")
    except (KeyError, IndexError, ValueError, OSError):
        return float("nan")
    value = header.get("B_DELTA")
    if value is None:
        return float("nan")
    if isinstance(value, str):
        return u.Quantity(value).to_value(u.deg)
    return float(value)


def summarize_runs(
    datacheck_file: Path,
    dl3_dir: Path,
    dl2_dir: Path | None = None,
    *,
    reduction: str | None = None,
) -> Table:
    """Read one row per selected run with a DL3 file; NSB comes from DL2 provenance."""
    selected = DataCheckTables.from_files([str(datacheck_file)]).statistics
    rows = []
    search_dl3_dir = dl3_dir
    if reduction is not None and (dl3_dir / reduction).is_dir():
        search_dl3_dir = dl3_dir / reduction

    for run in selected.to_dict(orient="records"):
        run_number = int(run["runnumber"])
        dl3_file = _find_run_file(search_dl3_dir, "dl3", run_number)
        if dl3_file is None and search_dl3_dir != dl3_dir:
            dl3_file = _find_run_file(dl3_dir, "dl3", run_number)
        if dl3_file is None:
            warnings.warn(f"No DL3 file for selected run {run_number}; skipped", stacklevel=2)
            continue

        metadata = fits.getheader(dl3_file, extname="EVENTS")
        nsb_level = float("nan")
        if dl2_dir is not None and dl2_dir.is_dir():
            dl2_file = _find_run_file(dl2_dir, "dl2", run_number)
            if dl2_file is not None:
                try:
                    nsb_level = LSTDL2EventTable(dl2_file).nsb_level or float("nan")
                except (KeyError, ValueError, OSError, AttributeError):
                    nsb_level = float("nan")

        ze_pnt = metadata.get("ZE_PNT")
        if ze_pnt is None:
            alt_pnt = metadata.get("ALT_PNT")
            ze_pnt = 90.0 - float(alt_pnt) if alt_pnt is not None else float("nan")

        rows.append(
            {
                "run_number": run_number,
                "date": int(run["date"]),
                "RA_PNT": float(metadata["RA_PNT"]),
                "DEC_PNT": float(metadata["DEC_PNT"]),
                "AZ_PNT": float(metadata["AZ_PNT"]),
                "ZE_PNT": float(ze_pnt),
                "B_DELTA": _b_delta_deg(dl3_file),
                "LIVETIME": float(metadata["LIVETIME"]),
                "N_EVENTS": int(metadata["NAXIS2"]),
                "NSB_LEVEL": nsb_level,
            }
        )

    table = Table(rows=rows, names=COLUMNS) if rows else Table(names=COLUMNS)
    for column in ANGLE_COLUMNS:
        table[column].unit = u.deg
    table["LIVETIME"].unit = u.s
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", nargs="?", type=Path, default=Path("config/config.toml"), help="Pipeline TOML file")
    parser.add_argument("-c", "--config", dest="config_option", type=Path, help="Pipeline TOML file")
    parser.add_argument(
        "-a",
        "--analysis-dir",
        "--analysis-directory",
        dest="analysis_dir",
        type=Path,
        default=Path.cwd(),
        help="init-lstana analysis directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--summary-file",
        dest="output",
        type=Path,
        help="Output summary table file (.ecsv or .csv; default: <analysis_dir>/run_summary.ecsv)",
    )
    parser.add_argument(
        "-r",
        "--reduction",
        type=str,
        help="DL3 reduction name to summarize (default: first configured reduction)",
    )
    parser.add_argument("--datacheck", type=Path, help="Override the configured datacheck selection file")
    parser.add_argument("--dl3-dir", type=Path, help="Override the configured DL3 directory")
    parser.add_argument("--dl2-dir", type=Path, help="Override the configured DL2 directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output table")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_file = args.config_option or args.config_file
    raw_config = load_config(config_file)
    pipeline_config = PipelineConfig.parse(raw_config)
    analysis_dir = args.analysis_dir.expanduser().resolve()
    _, dl2, dl3 = make_producers(pipeline_config, raw_config, analysis_dir)

    datacheck_file = args.datacheck or selection_file(raw_config, analysis_dir)

    if args.dl3_dir is not None:
        dl3_dir = args.dl3_dir.expanduser().resolve()
    elif args.reduction is not None:
        dl3_dir = (dl3.output_dir / args.reduction).resolve()
        if not dl3_dir.is_dir():
            raise FileNotFoundError(f"DL3 reduction directory not found: {dl3_dir}")
    elif (
        pipeline_config.dl3.enabled
        and pipeline_config.dl3.build is not None
        and pipeline_config.dl3.build.reductions
    ):
        first_reduction = pipeline_config.dl3.build.reductions[0].name
        first_reduction_dir = dl3.output_dir / first_reduction
        if first_reduction_dir.is_dir():
            dl3_dir = first_reduction_dir.resolve()
        else:
            dl3_dir = dl3.output_dir.resolve()
    else:
        dl3_dir = dl3.output_dir.resolve()

    dl2_dir = args.dl2_dir.expanduser().resolve() if args.dl2_dir else dl2.output_dir.resolve()

    if args.output is not None:
        summary_file = args.output.expanduser().resolve()
        if summary_file.is_dir():
            summary_file = summary_file / "run_summary.ecsv"
    else:
        summary_file = analysis_dir / "run_summary.ecsv"

    table = summarize_runs(datacheck_file, dl3_dir, dl2_dir, reduction=args.reduction)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    output_format = "ascii.csv" if summary_file.suffix.lower() == ".csv" else "ascii.ecsv"
    table.write(summary_file, format=output_format, overwrite=args.overwrite)
    print(f"Wrote {len(table)} runs to {summary_file}")


if __name__ == "__main__":
    main()
