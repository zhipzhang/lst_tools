from types import SimpleNamespace

import numpy as np
import pytest
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff
from regions import PointSkyRegion

from lst_tools.dl4.config import DL4DatasetConfig
from lst_tools.dl4.dataset_producer import DL4DatasetProducer


def test_spectrum_producer_writes_separate_rad_max_datasets(
    tmp_path,
    raw_config_factory,
    observation_factory,
    dl3_store_factory,
):
    dl3 = dl3_store_factory(tmp_path / "dl3", [observation_factory(2033), observation_factory(2034)])
    config = DL4DatasetConfig.parse(raw_config_factory(dl3_directory=dl3))

    summary = DL4DatasetProducer(config).produce()

    assert summary.counts == {"dl4:built": 2}
    assert [result.status for result in summary.results] == ["built", "built"]
    for obs_id, result in zip((2033, 2034), summary.results):
        assert result.output.name == f"dl4_LST-1.Run{obs_id:05d}_spectrum.fits"
        dataset = SpectrumDatasetOnOff.read(result.output)
        assert isinstance(dataset.counts.geom.region, PointSkyRegion)
        assert dataset.counts.data.sum() > 0
        assert dataset.counts_off.data.sum() > 0
        assert np.all(dataset.acceptance.data == 1)
        assert np.all(dataset.acceptance_off.data == 3)
        assert dataset.exposure is not None
        assert dataset.edisp is not None


def test_map_producer_writes_separate_datasets_without_background(
    tmp_path,
    raw_config_factory,
    observation_factory,
    dl3_store_factory,
):
    observations = [observation_factory(2033, point_like=False), observation_factory(2034, point_like=False)]
    dl3 = dl3_store_factory(tmp_path / "dl3", observations)
    config = DL4DatasetConfig.parse(raw_config_factory("map", dl3_directory=dl3))

    summary = DL4DatasetProducer(config).produce()

    assert summary.counts == {"dl4:built": 2}
    for obs_id, result in zip((2033, 2034), summary.results):
        assert result.output.name == f"dl4_LST-1.Run{obs_id:05d}_map.fits"
        dataset = MapDataset.read(result.output)
        assert dataset.counts.data.sum() > 0
        assert dataset.exposure is not None
        assert dataset.edisp is not None
        assert dataset.psf is not None
        assert dataset.background is None


def test_existing_outputs_are_preserved(tmp_path, raw_config_factory, observation_factory, dl3_store_factory):
    dl3 = dl3_store_factory(tmp_path / "dl3", [observation_factory()])
    config = DL4DatasetConfig.parse(raw_config_factory(dl3_directory=dl3))
    producer = DL4DatasetProducer(config)
    first = producer.produce()
    output = first.results[0].output
    modified_time = output.stat().st_mtime_ns

    second = producer.produce()

    assert second.counts == {"dl4:existing": 1}
    assert second.results[0].output == output
    assert output.stat().st_mtime_ns == modified_time


def test_failed_observation_does_not_stop_other_outputs(
    monkeypatch,
    tmp_path,
    raw_config_factory,
    observation_factory,
    dl3_store_factory,
):
    observations = [observation_factory(2033), observation_factory(2034)]
    dl3 = dl3_store_factory(tmp_path / "dl3", observations)
    config = DL4DatasetConfig.parse(raw_config_factory(dl3_directory=dl3))
    producer = DL4DatasetProducer(config)
    original_output_file = producer.output_file

    def fail_second(observation):
        if observation.obs_id == 2034:
            raise ValueError("missing EVENTS filename")
        return original_output_file(observation)

    monkeypatch.setattr(producer, "output_file", fail_second)

    with pytest.warns(UserWarning, match="observation 2034"):
        summary = producer.produce()

    assert summary.counts == {"dl4:built": 1, "dl4:failed": 1}
    assert summary.results[0].status == "built"
    assert summary.results[1].status == "failed"
    assert summary.results[0].output.exists()


def test_ds9_exclusion_is_rasterized_without_fits_input(
    raw_config_factory, observation_factory, dl3_store_factory, tmp_path
):
    dl3 = dl3_store_factory(tmp_path / "dl3", [observation_factory()])
    raw = raw_config_factory(dl3_directory=dl3)
    raw["dataset"]["spectrum"]["background"]["exclusion"] = {
        "regions": ["icrs;circle(83.632,22.0145,0.3)"],
        "mask_width_deg": 2.0,
        "mask_binsz_deg": 0.05,
    }
    producer = DL4DatasetProducer(DL4DatasetConfig.parse(raw))

    exclusion_mask = producer._exclusion_mask()

    assert not exclusion_mask.get_by_coord(producer.source_position)
    assert exclusion_mask.get_by_coord((80.0, 22.0145))


def test_output_name_preserves_fits_gz_stem(raw_config_factory, observation_factory, dl3_store_factory, tmp_path):
    dl3 = dl3_store_factory(tmp_path / "dl3", [observation_factory()])
    config = DL4DatasetConfig.parse(raw_config_factory(dl3_directory=dl3))
    producer = DL4DatasetProducer(config)
    observation = observation_factory(2033)
    producer.data_store = SimpleNamespace(
        hdu_table=SimpleNamespace(
            hdu_location=lambda **kwargs: SimpleNamespace(file_name="dl3_LST-1.Run02033.0001.fits.gz")
        )
    )

    assert producer.output_file(observation).name == "dl4_LST-1.Run02033.0001_spectrum.fits"


def test_no_selected_observations_is_an_error(raw_config_factory, observation_factory, dl3_store_factory, tmp_path):
    from gammapy.data import DataStore

    dl3 = dl3_store_factory(tmp_path / "dl3", [observation_factory()])
    raw = raw_config_factory(dl3_directory=dl3)
    raw["data"]["obs_ids"] = [9999]
    config = DL4DatasetConfig.parse(raw)
    data_store = DataStore.from_dir(dl3)

    with pytest.raises(ValueError, match="Missing obs_id"):
        DL4DatasetProducer(config, data_store=data_store).produce()
