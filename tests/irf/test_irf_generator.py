import json
from dataclasses import replace
from pathlib import Path

import pytest

from lst_tools.irf import GammaMCSample, IRFConfig
from lst_tools.irf.irf_generator import IRFGenerator


@pytest.fixture
def sample(tmp_path):
    dl2_path = tmp_path / "dl2_mc_merged.h5"
    dl2_path.touch()
    return GammaMCSample(
        path=dl2_path,
        mc_type="diffuse",
        nsb_level=0.81,
        dec_line=2276,
        zenith=10,
        azimuth=20,
    )


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
    return IRFConfig.from_cuts(80, 0.2, gh_cut=0.5)


def test_generate_writes_irf_config_and_log_alongside(tmp_path, sample, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    irf_file = generator.generate(sample)

    assert (
        irf_file == (tmp_path / "irfs/dec_2276/nsb_0.81/full-enclosure/ghcut0.5/theta_10_az_20.irf.fits.gz").resolve()
    )
    assert irf_file.is_file()

    config_file = irf_file.parent / "theta_10_az_20.config.json"
    with open(config_file) as file_handle:
        config = json.load(file_handle)
    assert config["EventSelector"]["filters"]["intensity"] == [80.0, float("inf")]
    assert config["DL3Cuts"]["global_gh_cut"] == pytest.approx(0.5)
    assert config["IRFFITSWriter"] == {"energy_dependent_gh": False, "point_like": False}
    assert "DataBinning" in config

    command = run_irf[0]
    assert f"--config={config_file}" in command
    assert f"--input-gamma-dl2={sample.path}" in command
    assert "--overwrite" in command


def test_generate_all_generates_all_requested_samples(tmp_path, sample, run_irf):
    samples = [sample, replace(sample, zenith=20)]

    irf_files = IRFGenerator(tmp_path / "irfs", fixed_config()).generate_all(samples)

    assert [path.name for path in irf_files] == [
        "theta_10_az_20.irf.fits.gz",
        "theta_20_az_20.irf.fits.gz",
    ]
    assert all(path.is_file() for path in irf_files)
    assert len(run_irf) == 2


def test_existing_irf_is_reused_when_the_config_is_unchanged(tmp_path, sample, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    first = generator.generate(sample)
    second = generator.generate(sample)

    assert first == second
    assert len(run_irf) == 1


def test_irf_is_regenerated_in_place_when_quality_cuts_change(tmp_path, sample, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())
    original_file = generator.generate(sample)

    new_config = IRFConfig.from_cuts(50, 0.2, gh_cut=0.5)
    updated_file = IRFGenerator(tmp_path / "irfs", new_config).generate(sample)

    assert updated_file == original_file
    assert len(run_irf) == 2
    with open(original_file.parent / f"{sample.pointing_name}.config.json") as file_handle:
        config = json.load(file_handle)
    assert config["EventSelector"]["filters"]["intensity"] == [50.0, float("inf")]


def test_point_like_config_uses_a_different_irf_file(tmp_path, sample, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())
    full_enclosure_file = generator.generate(sample)

    point_like_config = IRFConfig.from_cuts(80, 0.2, gh_cut=0.5, point_like=True, theta_cut=0.2)
    point_sample = replace(sample, mc_type="point-source")
    point_file = IRFGenerator(tmp_path / "irfs", point_like_config).generate(point_sample)

    assert point_file != full_enclosure_file
    assert point_file.parent.parent.name == "point"
    assert point_file.parent.name == "ghcut0.5_thetacut0.2"
    assert len(run_irf) == 2


def test_point_like_irf_rejects_diffuse_sample(tmp_path, sample, run_irf):
    config = IRFConfig.from_cuts(80, 0.2, gh_cut=0.5, point_like=True, theta_cut=0.2)
    generator = IRFGenerator(tmp_path / "irfs", config)

    with pytest.raises(ValueError, match="require a point-source"):
        generator.generate(sample)

    assert run_irf == []


def test_irf_of_another_gammaness_selection_gets_its_own_directory(tmp_path, sample, run_irf):
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    fixed_file = generator.generate(sample)
    efficiency_config = IRFConfig.from_cuts(80, 0.2, gh_efficiency=0.7)
    efficiency_file = IRFGenerator(tmp_path / "irfs", efficiency_config).generate(sample)

    assert fixed_file.parent.name == "ghcut0.5"
    assert efficiency_file.parent.name == "gheff0.7"
    assert len(run_irf) == 2


def test_raises_when_the_command_produces_no_file(tmp_path, sample, monkeypatch):
    monkeypatch.setattr("lst_tools.irf.irf_generator.subprocess.run", lambda command, check: None)
    generator = IRFGenerator(tmp_path / "irfs", fixed_config())

    with pytest.raises(RuntimeError, match="without creating"):
        generator.generate(sample)
