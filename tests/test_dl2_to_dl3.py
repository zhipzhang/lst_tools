import json
from collections import Counter

import pytest

from lst_tools.irf import IRFConfig
from lst_tools.pipeline.dl3_build import DL3Build
from lst_tools.scripts import dl2_to_dl3


def test_main_loads_irf_config_and_forwards_options(tmp_path, monkeypatch):
    irf_config = IRFConfig.from_cuts(80, 1.0, gh_cut=0.7)
    config_path = irf_config.write_irf_config(tmp_path / "irf_config.json")
    dl2_file = tmp_path / "dl2_LST-1.Run10623.h5"
    dl2_file.touch()
    calls = []

    def fake_run(self, max_workers=None, runs_per_worker=5):
        calls.append((self, max_workers, runs_per_worker))
        return Counter({"dl3:reduced": 1})

    monkeypatch.setattr(DL3Build, "run", fake_run)

    counts = dl2_to_dl3.main(
        [
            "--input",
            str(dl2_file),
            "--output-dir",
            str(tmp_path / "dl3"),
            "--irf-dir",
            str(tmp_path / "irfs"),
            "--irf-config",
            str(config_path),
            "--mc-dir",
            "/mc/nsb_$NSB_LEVEL",
            "--source-name",
            "Crab",
            "--source-ra",
            "83.6",
            "--source-dec",
            "22.0",
            "--interp-method",
            "nearest",
            "--use-nearest-irf-node",
            "--overwrite",
            "--max-workers",
            "3",
            "--runs-per-worker",
            "4",
        ]
    )

    assert counts == {"dl3:reduced": 1}
    assert len(calls) == 1
    build, max_workers, runs_per_worker = calls[0]
    assert build.dl2_files == (dl2_file.resolve(),)
    assert build.mc_dl2_directory == "/mc/nsb_$NSB_LEVEL"
    assert build.reducer.output_dir == (tmp_path / "dl3").resolve()
    assert build.reducer.irfs_root == (tmp_path / "irfs").resolve()
    assert build.reducer.irf_config.to_irf_config() == irf_config.to_irf_config()
    assert build.reducer.source_name == "Crab"
    assert build.reducer.source_ra == 83.6
    assert build.reducer.source_dec == 22.0
    assert build.reducer.interp_method == "nearest"
    assert build.reducer.use_nearest_irf_node is True
    assert build.reducer.overwrite is True
    assert max_workers == 3
    assert runs_per_worker == 4


def test_worker_rejects_missing_irf_config(tmp_path):
    with pytest.raises(SystemExit):
        dl2_to_dl3.main(
            [
                "--input",
                str(tmp_path / "dl2.h5"),
                "--output-dir",
                str(tmp_path / "dl3"),
                "--irf-dir",
                str(tmp_path / "irfs"),
                "--irf-config",
                str(tmp_path / "missing.json"),
                "--mc-dir",
                "/mc",
                "--source-name",
                "Crab",
                "--source-ra",
                "83.6",
                "--source-dec",
                "22.0",
            ]
        )


@pytest.mark.parametrize("flag", ["--max-workers", "--runs-per-worker"])
def test_worker_rejects_nonpositive_limits(tmp_path, flag):
    config_path = IRFConfig.from_cuts(80, 1.0, gh_cut=0.7).write_irf_config(tmp_path / "irf_config.json")
    with pytest.raises(SystemExit):
        dl2_to_dl3.main(
            [
                "--input",
                str(tmp_path / "dl2.h5"),
                "--output-dir",
                str(tmp_path / "dl3"),
                "--irf-dir",
                str(tmp_path / "irfs"),
                "--irf-config",
                str(config_path),
                "--mc-dir",
                "/mc",
                "--source-name",
                "Crab",
                "--source-ra",
                "83.6",
                "--source-dec",
                "22.0",
                flag,
                "0",
            ]
        )


def test_build_reuses_products_with_real_processes(tmp_path):
    import h5py
    import numpy as np
    import pandas as pd
    from astropy.io import fits
    from ctapipe.containers import EventType
    from lstchain.io.io import dl2_params_lstcam_key
    from lstchain.paths import dl2_to_dl3_filename

    from lst_tools.irf import GammaMCSample, irf_file_path
    from lst_tools.pipeline import DL3BuildOptions, DL3Producer, DL3ProducerConfig, DL3ReductionOptions, RunSelection
    from lst_tools.pipeline.configs import SourceConfig

    dl2_dir = tmp_path / "dl2"
    dl2_dir.mkdir(parents=True)
    dl2_file = dl2_dir / "dl2_LST-1.Run10623.h5"
    events = pd.DataFrame({
        "event_type": [EventType.SUBARRAY.value] * 4,
        "dragon_time": 1_700_000_000 + np.arange(4) * 0.001,
        "delta_t": [0.0001, 0.0002, 0.0003, 0.0004],
        "alt_tel": [1.0] * 4,
        "az_tel": [2.0] * 4,
        "trigger_time": 1_700_000_000 + np.arange(4) * 0.001,
    })
    events.to_hdf(dl2_file, key=dl2_params_lstcam_key)
    with h5py.File(dl2_file, "a") as file_handle:
        file_handle.create_dataset("provenance/dl1_to_dl2", data=json.dumps({
            "input": [{"role": "model", "url": "/models/nsb_tuning_0.38/dec_347"}]
        }))
    (dl2_dir / "dl2_LST-1.Run99999.h5").touch()

    reductions = tuple(
        DL3ReductionOptions(
            name,
            IRFConfig.from_cuts(
                intensity,
                1.0,
                gh_cut=0.7,
                point_like=point,
                theta_cut=0.2 if point else None,
            ),
        )
        for name, intensity, point in (("standard", 80, False), ("point", 150, True))
    )
    campaign = tmp_path / "mc_nsb_0.38"
    for reduction in reductions:
        config = reduction.irf_config
        mc_type = "Gamma" if config.point_like else "GammaDiffuse"
        mc_file = campaign / "TestingDataset" / mc_type / "dec_347" / "node_theta_20_az_180" / "dl2_test_merged.h5"
        mc_file.parent.mkdir(parents=True, exist_ok=True)
        mc_file.touch()
        sample = GammaMCSample(
            path=mc_file,
            mc_type="point-source" if config.point_like else "diffuse",
            nsb_level=0.38,
            dec_line=347,
            zenith=20,
            azimuth=180,
        )
        relative_path = irf_file_path(sample, irf_type=config.irf_type, cut_label=config.cut_label)
        irf_file = tmp_path / "irfs" / reduction.name / relative_path
        irf_file.parent.mkdir(parents=True)
        fits.PrimaryHDU().writeto(irf_file)
        (irf_file.parent / f"{sample.pointing_name}.config.json").write_text(json.dumps(config.to_irf_config()))
        output = tmp_path / "dl3" / reduction.name / dl2_to_dl3_filename(dl2_file)
        output.parent.mkdir(parents=True)
        fits.PrimaryHDU().writeto(output)

    selection = RunSelection(pd.DataFrame([{"runnumber": 10623, "date": 20220426}]))
    producer = DL3Producer(
        DL3ProducerConfig(enabled=True, source="build", build=DL3BuildOptions(
            reductions=reductions, mc_dl2_directory=str(tmp_path / "mc_nsb_$NSB_LEVEL")
        )),
        tmp_path,
        source=SourceConfig("Crab", 83.6, 22.0),
    )

    assert producer.produce(selection) == {
        "dl3:irf_files": 2, "dl3:reduced": 2, "dl3:failed": 0
    }
    assert not list((tmp_path / "dl3").rglob("*Run99999*"))
