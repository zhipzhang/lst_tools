import pytest
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff

from lst_tools.scripts import generate_dl4


@pytest.mark.parametrize("dataset_type", ["spectrum", "map"])
def test_generate_dl4_cli_end_to_end(
    monkeypatch,
    tmp_path,
    capsys,
    dataset_type,
    raw_config_factory,
    observation_factory,
    dl3_store_factory,
):
    observation = observation_factory(point_like=dataset_type == "spectrum")
    dl3 = dl3_store_factory(tmp_path / "dl3", [observation])
    raw = raw_config_factory(dataset_type, dl3_directory=dl3)
    config_path = tmp_path / "dl4.toml"
    config_path.write_text("loaded through the test monkeypatch\n")
    monkeypatch.setattr(generate_dl4, "load_config", lambda path: raw)
    monkeypatch.setattr("sys.argv", ["generate-dl4", str(config_path)])

    generate_dl4.main()

    output = tmp_path / "dl4" / f"dl4_LST-1.Run{observation.obs_id:05d}_{dataset_type}.fits"
    assert output.exists()
    assert "2033: built" in capsys.readouterr().out
    if dataset_type == "spectrum":
        assert SpectrumDatasetOnOff.read(output).counts_off is not None
    else:
        assert MapDataset.read(output).background is None


def test_generate_dl4_directory_overrides(
    monkeypatch,
    tmp_path,
    raw_config_factory,
    observation_factory,
    dl3_store_factory,
):
    dl3 = dl3_store_factory(tmp_path / "actual_dl3", [observation_factory(point_like=False)])
    raw = raw_config_factory("map", dl3_directory=tmp_path / "wrong_dl3")
    raw["output"]["directory"] = str(tmp_path / "wrong_dl4")
    output = tmp_path / "actual_dl4"
    monkeypatch.setattr(generate_dl4, "load_config", lambda path: raw)
    monkeypatch.setattr(
        "sys.argv",
        ["generate-dl4", "config.toml", "--input-directory", str(dl3), "--output-directory", str(output)],
    )

    generate_dl4.main()

    assert (output / "dl4_LST-1.Run02033_map.fits").exists()
    assert not (tmp_path / "wrong_dl4").exists()


def test_generate_dl4_requires_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate-dl4"])

    with pytest.raises(SystemExit):
        generate_dl4.main()
