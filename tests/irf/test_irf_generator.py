import json
from pathlib import Path

import pytest

from lst_tools.event_filter import EventFilter
from lst_tools.irf import IRFConfig, IRFNode
from lst_tools.irf.irf_generator import IRFGenerator


@pytest.fixture
def node(tmp_path):
    dl2_path = tmp_path / "dl2_mc_merged.h5"
    dl2_path.touch()
    return IRFNode(nsb_level=0.81, dec_line=2276, zenith=10, azimuth=20, dl2_path=dl2_path)


@pytest.fixture
def run_irf(monkeypatch):
    """Mock lstchain_create_irf_files; records commands, creates the output file."""
    calls = []

    def run(command, check):
        calls.append(command)
        output_argument = next(argument for argument in command if argument.startswith("--output-irf-file="))
        Path(output_argument.split("=", 1)[1]).touch()

    monkeypatch.setattr("lst_tools.irf.irf_generator.subprocess.run", run)
    return calls


def fixed_config():
    return IRFConfig(event_filter=EventFilter.from_cuts(80, 0.2, gh_cut=0.5))


def test_make_irf_node_writes_irf_config_and_log_alongside(tmp_path, node, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    irf_file = generator.make_irf_node(node)

    assert irf_file == (tmp_path / "irfs/nsb_0.81/dec_2276/theta_10_az_20.irf.fits.gz").resolve()
    assert irf_file.is_file()

    config_file = irf_file.parent / "theta_10_az_20.config.json"
    with open(config_file) as file_handle:
        config = json.load(file_handle)
    assert config["EventSelector"]["filters"]["intensity"] == [80.0, float("inf")]
    assert config["DL3Cuts"]["global_gh_cut"] == pytest.approx(0.5)
    assert config["IRFFITSWriter"] == {"energy_dependent_gh": False}
    assert "DataBinning" in config

    command = run_irf[0]
    assert f"--config={config_file}" in command
    assert f"--input-gamma-dl2={node.dl2_path}" in command
    assert "--overwrite" in command


def test_existing_irf_is_reused_when_the_config_is_unchanged(tmp_path, node, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    first = generator.make_irf_node(node)
    second = generator.make_irf_node(node)

    assert first == second
    assert len(run_irf) == 1


def test_irf_is_regenerated_when_the_config_changes(tmp_path, node, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())
    generator.make_irf_node(node)

    new_config = IRFConfig(event_filter=EventFilter.from_cuts(80, 0.2, gh_cut=0.9))
    IRFGenerator(tmp_path / "irfs", new_config).make_irf_node(node)

    assert len(run_irf) == 2


def test_quantile_mode_marks_energy_dependent_gh_in_the_config(tmp_path, node, run_irf):
    irf_config = IRFConfig(event_filter=EventFilter.from_cuts(80, 0.2, gh_efficiency=0.7))
    generator = IRFGenerator(tmp_path / "irfs", irf_config)

    irf_file = generator.make_irf_node(node)

    with open(irf_file.parent / "theta_10_az_20.config.json") as file_handle:
        config = json.load(file_handle)
    assert config["IRFFITSWriter"] == {"energy_dependent_gh": True}
    assert config["DL3Cuts"]["gh_efficiency"] == pytest.approx(0.7)


def test_raises_when_the_command_produces_no_file(tmp_path, node, monkeypatch):
    monkeypatch.setattr("lst_tools.irf.irf_generator.subprocess.run", lambda command, check: None)
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    with pytest.raises(RuntimeError, match="without creating"):
        generator.make_irf_node(node)


def test_irf_config_serializes_all_lstchain_sections():
    config = fixed_config().to_lstchain_config()

    assert set(config) == {"EventSelector", "DL3Cuts", "IRFFITSWriter", "DataBinning"}
    assert config["DataBinning"]["fov_offset_min"] == pytest.approx(0)
    assert config["DataBinning"]["fov_offset_max"] == pytest.approx(2.5)
    assert config["DataBinning"]["fov_offset_n_edges"] == 6


def test_default_irf_config_reproduces_default_config_json():
    import lst_tools.irf

    with open(Path(lst_tools.irf.__file__).parent / "default_config.json") as file_handle:
        reference = json.load(file_handle)

    config = IRFConfig(event_filter=EventFilter.from_cuts(50, 1.0, gh_cut=0.7)).to_lstchain_config()

    assert config == {**reference, "IRFFITSWriter": {"energy_dependent_gh": False}}
