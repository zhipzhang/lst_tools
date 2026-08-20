import numpy as np
import pandas as pd
import pytest

from lst_ulities.datacheck import RunStatistics
from lst_ulities.dl3 import parse_dl3_path
from lst_ulities.scripts.init_lstana import create_data_links, create_dl3_links


def test_create_data_links_groups_runs_and_is_idempotent(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    source_files = {}
    for level in ("dl1", "dl2"):
        source = sources / f"{level}_LST-1.Run00001.h5"
        source.touch()
        source_files[level] = source

    stats = RunStatistics(
        pd.DataFrame(
            {
                "date": [20240101],
                "mean_cos_zd": [np.cos(np.radians(15.0))],
            },
            index=[1],
        )
    ).assign_zenith_bins([0, 20, 40])

    def find_path(date, run_number, level):
        assert date == 20240101
        assert run_number == 1
        return str(source_files[level])

    output = tmp_path / "analysis"
    first = create_data_links(stats, output, ("dl1", "dl2"), path_finder=find_path)
    second = create_data_links(stats, output, ("dl1", "dl2"), path_finder=find_path)

    assert first == {"dl1:created": 1, "dl2:created": 1}
    assert second == {"dl1:existing": 1, "dl2:existing": 1}
    for level, source in source_files.items():
        destination = output / level / "zd_0_20" / source.name
        assert destination.is_symlink()
        assert destination.resolve() == source.resolve()


def test_create_data_links_does_not_replace_conflicting_file(tmp_path):
    stats = RunStatistics(
        pd.DataFrame({"date": [20240101], "mean_cos_zd": [1.0]}, index=[1])
    ).assign_zenith_bins([0, 20])
    destination_dir = tmp_path / "dl1" / "zd_0_20"
    destination_dir.mkdir(parents=True)
    destination = destination_dir / "dl1_LST-1.Run00001.h5"
    destination.write_text("keep me")

    def find_path(date, run_number, level):
        return str(tmp_path / "source" / destination.name)

    with pytest.warns(UserWarning, match="Refusing to replace"):
        counts = create_data_links(stats, tmp_path, ("dl1",), path_finder=find_path)

    assert counts == {"dl1:conflict": 1}
    assert destination.read_text() == "keep me"


def test_create_data_links_rejects_unsupported_level(tmp_path):
    stats = RunStatistics(pd.DataFrame(columns=["date", "zenith_bin"]))

    with pytest.raises(ValueError, match="Unsupported data levels"):
        create_data_links(stats, tmp_path, ("dl3",))


def test_create_dl3_links_uses_only_configured_products(tmp_path):
    stats = RunStatistics(
        pd.DataFrame({"date": [20240101], "mean_cos_zd": [1.0]}, index=[1])
    ).assign_zenith_bins([0, 20])
    dl3_config = {
        "cut_configs": ["gheff0.7_thetacont0.7", "gheff0.9_thetacont0.7"],
        "products": [
            {
                "name": "point",
                "analysis_type": "point",
                "background_type": "ring-wobble",
            }
        ],
    }
    available = []
    for cut_config in [
        "gheff0.7_thetacont0.7",
        "gheff0.9_thetacont0.7",
        "gheff0.5_thetacont0.7",
    ]:
        source = tmp_path / "source" / "point" / "ring-wobble" / cut_config / "irf_interp" / "run.fits"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        available.append(parse_dl3_path(source, 20240101, 1))

    def find_products(date, run_number):
        assert (date, run_number) == (20240101, 1)
        return available

    first = create_dl3_links(stats, tmp_path / "output", dl3_config, product_finder=find_products)
    second = create_dl3_links(stats, tmp_path / "output", dl3_config, product_finder=find_products)

    assert sum(value for key, value in first.items() if key.endswith(":created")) == 2
    assert sum(value for key, value in second.items() if key.endswith(":existing")) == 2
    for cut_config in dl3_config["cut_configs"]:
        destination = tmp_path / "output" / "dl3" / "point" / cut_config / "zd_0_20" / "run.fits"
        assert destination.is_symlink()
    unconfigured = (
        tmp_path
        / "output"
        / "dl3"
        / "point"
        / "gheff0.5_thetacont0.7"
        / "zd_0_20"
        / "run.fits"
    )
    assert not unconfigured.exists()
