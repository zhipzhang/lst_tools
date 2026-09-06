from pathlib import Path

import pytest

from lst_tools.irf import IRFNode
from lst_tools.irf.irf_generator import IRFGenerator


def make_irf_node_path(tmp_path, node: IRFNode):
    irf_path = tmp_path / node.path_name
    irf_path.mkdir(parents=True)
    return irf_path


def make_irf_node_file(irf_path, node: IRFNode):
    irf_path_file = irf_path / node.path_name / "irf.fits.gz"
    irf_path_file.touch()
    return irf_path_file


def test_non_existing_irf_node(tmp_path):
    irf_generator = IRFGenerator(tmp_path)
    node = IRFNode(declination=22.76, azimuth=10, zenith=10, intensity_cuts=70, dl2_path=None, gh_efficiency=0.7)
    assert not irf_generator._irf_node_exist(node)
    make_irf_node_path(tmp_path, node)
    with pytest.warns(UserWarning):
        assert not irf_generator._irf_node_exist(node)
    make_irf_node_file(tmp_path, node)
    assert irf_generator._irf_node_exist(node)


def test_make_irf_node_creates_directory_and_reuses_output(tmp_path, monkeypatch):
    calls = []

    def run(command, check):
        calls.append((command, check))
        output_argument = next(argument for argument in command if argument.startswith("--output-irf-file="))
        Path(output_argument.split("=", 1)[1]).touch()

    monkeypatch.setattr("lst_tools.irf.irf_generator.subprocess.run", run)
    dl2_path = tmp_path / "mc_dl2.h5"
    dl2_path.touch()
    node = IRFNode(
        declination=22.76,
        azimuth=10,
        zenith=20,
        intensity_cuts=70,
        dl2_path=dl2_path,
        gh_efficiency=0.7,
    )
    generator = IRFGenerator(tmp_path / "irfs")

    first = generator.make_irf_node(node)
    second = generator.make_irf_node(node)

    assert first == second
    assert first.exists()
    assert (first.parent / "config.json").exists()
    assert len(calls) == 1
