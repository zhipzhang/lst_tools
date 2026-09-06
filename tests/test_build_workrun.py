from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lst_tools.irf import IRFNode
from lst_tools.scripts import build_workrun
from lst_tools.scripts.build_workrun import (
    BuildWorkRun,
    OffRunSelectionBounds,
    expected_dl2_path,
    find_compatible_offrun_dl1_files,
    find_tailcuts_level,
    find_target_dl2_file,
    reconstruct_off_runs,
    seasonal_day_of_year,
    select_matching_off_runs,
    write_workrun_config,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/data/tailcut1005/dl1.h5", (10, 5)),
        ("/data/tailcut84/dl1.h5", (8, 4)),
        ("/data/tailcut105/dl1.h5", None),
        ("/data/no_cleaning/dl1.h5", None),
    ],
)
def test_find_tailcuts_level(path, expected):
    assert find_tailcuts_level(path) == expected


def test_find_target_dl2_file_accepts_a_file_or_directory(tmp_path):
    target = tmp_path / "nested" / "dl2_LST-1.Run00042.h5"
    target.parent.mkdir()
    target.touch()

    assert find_target_dl2_file(target, 42) == target.resolve()
    assert find_target_dl2_file(tmp_path, 42) == target.resolve()


def test_select_matching_off_runs_excludes_target():
    cos_30_deg = np.cos(np.radians(30))
    statistics = pd.DataFrame(
        {
            "run_number": [42, 43, 44, 45],
            "date": [20240215, 20240301, 20240301, 20241001],
            "mean_cos_zd": [cos_30_deg, np.cos(np.radians(31)), np.cos(np.radians(40)), cos_30_deg],
            "mean_diffuse_nsb_std": [2.0, 2.1, 2.0, 2.0],
        }
    )

    selected, bounds = select_matching_off_runs(
        statistics,
        42,
        nsb_relative_tolerance=0.1,
        zenith_tolerance_deg=2,
        date_tolerance_days=90,
    )

    assert selected["run_number"].tolist() == [43]
    assert bounds.nsb_min == pytest.approx(1.8)
    assert bounds.nsb_max == pytest.approx(2.2)
    assert bounds.zenith_min_deg == pytest.approx(28)
    assert bounds.zenith_max_deg == pytest.approx(32)


def test_seasonal_date_selection_uses_calendar_days_and_wraps_new_year():
    statistics = pd.DataFrame(
        {
            "run_number": [42, 43, 44, 45],
            "date": [20241231, 20250101, 20240301, 20240930],
            "mean_cos_zd": [0.8, 0.8, 0.8, 0.8],
            "mean_diffuse_nsb_std": [2.0, 2.0, 2.0, 2.0],
        }
    )

    selected, bounds = select_matching_off_runs(
        statistics,
        42,
        nsb_relative_tolerance=0.1,
        zenith_tolerance_deg=2,
        date_tolerance_days=90,
    )

    assert selected["run_number"].tolist() == [43, 44]
    assert bounds.target_day_of_year == 366
    assert bounds.date_tolerance_days == 90


def test_seasonal_day_of_year_uses_a_common_leap_year():
    result = seasonal_day_of_year(pd.Series([20240228, 20240229, 20250301, 20251301]))

    assert result.iloc[:3].tolist() == [59, 60, 61]
    assert np.isnan(result.iloc[3])


def test_find_compatible_offrun_files_checks_resolved_tailcuts(tmp_path):
    compatible_source = tmp_path / "source" / "tailcut1005" / "dl1_LST-1.Run00043.h5"
    incompatible_source = tmp_path / "source" / "tailcut84" / "dl1_LST-1.Run00044.h5"
    compatible_source.parent.mkdir(parents=True)
    incompatible_source.parent.mkdir(parents=True)
    compatible_source.touch()
    incompatible_source.touch()

    links = tmp_path / "offruns"
    links.mkdir()
    (links / compatible_source.name).symlink_to(compatible_source)
    (links / incompatible_source.name).symlink_to(incompatible_source)

    with pytest.warns(UserWarning, match="Tailcuts mismatch"):
        result = find_compatible_offrun_dl1_files(links, [43, 44], (10, 5))

    assert result == [compatible_source.resolve()]


def test_reconstruct_off_runs_uses_model_and_skips_existing_outputs(tmp_path, monkeypatch):
    first = tmp_path / "dl1_LST-1.Run00043.h5"
    second = tmp_path / "dl1_LST-1.Run00044.h5"
    first.touch()
    second.touch()
    output_dir = tmp_path / "offdl2"
    output_dir.mkdir()
    existing = expected_dl2_path(first, output_dir)
    existing.touch()
    calls = []

    class FakeDL1ToDL2Tool:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.input_files = kwargs["input_files"]
            self.output_dir = kwargs["output_dir"]

        def setup(self):
            calls.append("setup")

        def start(self):
            calls.append("start")
            for input_file in self.input_files:
                expected_dl2_path(input_file, self.output_dir).touch()

    monkeypatch.setattr(build_workrun, "DL1ToDL2Tool", FakeDL1ToDL2Tool)
    model_directory = tmp_path / "models"

    outputs = reconstruct_off_runs([first, second], model_directory, output_dir)

    assert outputs == [existing, expected_dl2_path(second, output_dir)]
    assert calls[0]["input_files"] == [second]
    assert calls[0]["path_models"] == model_directory
    assert calls[0]["output_dir"] == output_dir
    assert calls[1:] == ["setup", "start"]


def test_write_workrun_config_produces_valid_toml(tmp_path):
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    config_file = tmp_path / "workrun.toml"
    write_workrun_config(
        config_file,
        {
            "target": {"run_number": 42, "dl2_file": Path("/data/run 42.h5"), "tailcuts": [10, 5]},
            "irf": {"gh_efficiency": 0.7, "linked_files": [Path("/irf/one.fits.gz")]},
        },
    )

    with config_file.open("rb") as file_handle:
        config = tomllib.load(file_handle)

    assert config["target"] == {
        "run_number": 42,
        "dl2_file": "/data/run 42.h5",
        "tailcuts": [10, 5],
    }
    assert config["irf"]["gh_efficiency"] == pytest.approx(0.7)
    assert config["irf"]["linked_files"] == ["/irf/one.fits.gz"]


def test_build_workrun_start_creates_links_and_config(tmp_path, monkeypatch):
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    target_dl2 = tmp_path / "inputs" / "dl2_LST-1.Run00042.h5"
    mc_dl2 = tmp_path / "mc" / "dl2_gamma_merged.h5"
    target_dl2.parent.mkdir()
    mc_dl2.parent.mkdir()
    target_dl2.touch()
    mc_dl2.touch()

    def run_irf(command, check):
        assert check is True
        output_argument = next(argument for argument in command if argument.startswith("--output-irf-file="))
        Path(output_argument.split("=", 1)[1]).touch()

    monkeypatch.setattr("lst_tools.irf.irf_generator.subprocess.run", run_irf)

    tool = BuildWorkRun(
        run_number=42,
        output_dir=tmp_path / "workdir",
        irf_output_dir=tmp_path / "shared_irfs",
        mc_dl2_path=tmp_path / "mc",
    )
    tool.target_dl2_file = target_dl2.resolve()
    tool.target_dl2_table = SimpleNamespace(
        model_directory=str(tmp_path / "models"),
        tailcut_level=(10, 5),
        nsb_level=0.81,
        dec_line=2276,
        intensity_cuts=80,
    )
    tool.matching_off_runs = pd.DataFrame({"run_number": []}, dtype=int)
    tool.selection_bounds = OffRunSelectionBounds(0.72, 0.90, 28, 32, 46, 90)
    tool.offrun_dl1_files = []
    tool.irf_nodes = [
        IRFNode(
            declination=22.76,
            zenith=30,
            azimuth=120,
            intensity_cuts=80,
            dl2_path=mc_dl2,
            gh_efficiency=0.7,
        )
    ]
    tool.workrun_dir = Path(tool.output_dir).resolve() / "Run00042"
    tool.offrun_dl2_dir = tool.workrun_dir / "offdl2"

    tool.start()

    target_link = tool.workrun_dir / target_dl2.name
    assert target_link.resolve() == target_dl2.resolve()
    irf_node_link = tool.workrun_dir / "irf" / tool.irf_nodes[0].pointing_name
    assert irf_node_link.is_symlink()
    assert irf_node_link.resolve() == (Path(tool.irf_output_dir) / tool.irf_nodes[0].path_name).resolve()
    assert (irf_node_link / "irf.fits.gz").is_file()
    with (tool.workrun_dir / "workrun.toml").open("rb") as file_handle:
        config = tomllib.load(file_handle)
    assert config["target"]["run_number"] == 42
    assert config["target"]["tailcuts"] == [10, 5]
    assert config["irf"]["node_count"] == 1
    assert config["irf"]["linked_nodes"] == [str(irf_node_link)]
    assert config["offrun_selection"]["date_tolerance_days"] == 90
    assert config["offrun_selection"]["target_day_of_year"] == 46
