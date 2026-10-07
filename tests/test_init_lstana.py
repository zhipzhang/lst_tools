import numpy as np
import pandas as pd
import pytest

from lst_tools.datacheck import assign_zenith_bins
from lst_tools.scripts.init_lstana import (
    create_data_links,
    load_irf_config,
    prepare_working_directory,
    validate_config,
)


def test_prepare_working_directory_creates_the_stage_layout(tmp_path):
    config = {
        "dl2": {"rf_directory": "/models/nsb_tuning_0.24/dec_347"},
        "dl3": {"enabled": True},
    }

    prepare_working_directory(tmp_path, (0.0, 20.0, 40.0), config)

    assert (tmp_path / "data_check").is_dir()
    for level in ("dl1", "dl2", "dl3"):
        assert (tmp_path / level / "zd_0_20").is_dir()
        assert (tmp_path / level / "zd_20_40").is_dir()


def test_prepare_working_directory_only_creates_dirs_of_active_stages(tmp_path):
    prepare_working_directory(tmp_path, (0.0, 20.0), {})

    assert (tmp_path / "data_check").is_dir()
    assert (tmp_path / "dl1" / "zd_0_20").is_dir()
    assert not (tmp_path / "dl2").exists()
    assert not (tmp_path / "dl3").exists()


def test_validate_config_requires_rf_directory_with_mc_group_naming():
    with pytest.raises(ValueError, match="rf_directory"):
        validate_config({"dl2": {}})
    with pytest.raises(ValueError, match="nsb_tuning_<x>/dec_<NNNN>"):
        validate_config({"dl2": {"rf_directory": "/models/somewhere"}})
    validate_config({"dl2": {"rf_directory": "/models/nsb_tuning_0.24/dec_347"}})


def test_validate_config_dl3_requires_the_irf_section():
    with pytest.raises(ValueError, match=r"\[irf\] section"):
        validate_config({"dl3": {"enabled": True}})

    validate_config(
        {
            "dl3": {"enabled": True},
            "irf": {"intensity_cut": 80, "leakage_cut": 1.0, "gh_cut": 0.7},
        }
    )


def test_load_irf_config_builds_the_shared_event_filter():
    config = {
        "irf": {
            "intensity_cut": 80,
            "leakage_cut": 1.0,
            "gh_efficiency": 0.7,
        }
    }

    irf_config = load_irf_config(config)

    event_selector = irf_config.event_filter.event_selector
    assert event_selector.filters["intensity"] == [80.0, np.inf]
    assert event_selector.filters["leakage_intensity_width_2"] == [0, 1.0]
    assert irf_config.event_filter.energy_dependent_gh is True
    assert irf_config.event_filter.dl3_cuts.gh_efficiency == 0.7


def test_load_irf_config_requires_the_irf_section():
    with pytest.raises(ValueError, match=r"\[irf\] section"):
        load_irf_config({})


def test_load_irf_config_requires_exactly_one_gammaness_mode():
    section = {"intensity_cut": 80, "leakage_cut": 1.0}

    with pytest.raises(ValueError, match="exactly one of gh_cut and gh_efficiency"):
        load_irf_config({"irf": section})
    with pytest.raises(ValueError, match="exactly one of gh_cut and gh_efficiency"):
        load_irf_config({"irf": {**section, "gh_cut": 0.7, "gh_efficiency": 0.7}})


def test_create_data_links_groups_runs_and_is_idempotent(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    source_files = {}
    for level in ("dl1", "dl2"):
        source = sources / f"{level}_LST-1.Run00001.h5"
        source.touch()
        source_files[level] = source

    stats = assign_zenith_bins(
        pd.DataFrame(
            {
                "run_number": [1],
                "date": [20240101],
                "mean_cos_zd": [np.cos(np.radians(15.0))],
            }
        ),
        [0, 20, 40],
    )

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
    stats = assign_zenith_bins(
        pd.DataFrame({"run_number": [1], "date": [20240101], "mean_cos_zd": [1.0]}),
        [0, 20],
    )
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
    stats = pd.DataFrame(columns=["run_number", "date", "zenith_bin"])

    with pytest.raises(ValueError, match="Unsupported data levels"):
        create_data_links(stats, tmp_path, ("dl3",))
