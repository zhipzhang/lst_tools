import copy
from pathlib import Path

import pytest
import tomllib

from lst_tools.dl4.config import DL4ConfigError, DL4DatasetConfig


def test_shipped_example_parses_and_builds_energy_axes():
    with Path("config/dl4_datasets.toml").open("rb") as file_handle:
        raw = tomllib.load(file_handle)

    with pytest.warns(UserWarning, match="dataset.map"):
        config = DL4DatasetConfig.parse(raw)

    assert config.data.required_irf == ("aeff", "edisp", "rad_max")
    assert config.dataset.type == "spectrum"
    assert config.dataset.energy.reconstructed.to_axis("energy").nbin == 30
    assert config.dataset.energy.true.to_axis("energy_true").nbin > 60
    assert config.dataset.spectrum.background.exclusion.regions == ()


def test_map_configuration(raw_config_factory):
    config = DL4DatasetConfig.parse(raw_config_factory("map"))

    assert config.dataset.type == "map"
    assert config.dataset.spectrum is None
    assert config.dataset.map.maker.selection == ("counts", "exposure", "edisp", "psf")
    assert config.dataset.map.geometry.width_deg == (2.0, 2.0)


def test_spectrum_maker_and_exclusion_options(raw_config_factory):
    raw = raw_config_factory()
    raw["dataset"]["spectrum"]["maker"]["containment_correction"] = True
    raw["data"]["required_irf"].append("psf")
    raw["dataset"]["spectrum"]["background"]["exclusion"] = {
        "regions": ["galactic;circle(183.604,-8.708,0.5)"],
        "mask_width_deg": 8.0,
        "mask_binsz_deg": 0.02,
    }

    config = DL4DatasetConfig.parse(raw)

    assert config.dataset.spectrum.maker.containment_correction is True
    exclusion = config.dataset.spectrum.background.exclusion
    assert len(exclusion.sky_regions()) == 1
    assert exclusion.sky_regions()[0].center.frame.name == "galactic"
    assert exclusion.mask_binsz_deg == 0.02


def test_safe_mask_options(raw_config_factory):
    raw = raw_config_factory()
    raw["dataset"]["safe_mask"] = {
        "enabled": True,
        "methods": ["aeff-max", "edisp-bias"],
        "aeff_percent": 15.0,
        "bias_percent": 5.0,
        "offset_max_deg": 2.0,
        "irfs": "DL3",
        "fixed_offset_deg": 0.5,
    }

    options = DL4DatasetConfig.parse(raw).dataset.safe_mask

    assert options.methods == ("aeff-max", "edisp-bias")
    assert options.aeff_percent == 15.0
    assert options.irfs == "DL3"
    assert options.fixed_offset_deg == 0.5


def test_inactive_dataset_options_warn(raw_config_factory):
    raw = raw_config_factory()
    raw["dataset"]["map"] = {"invalid": True}

    with pytest.warns(UserWarning, match="ignored"):
        config = DL4DatasetConfig.parse(raw)

    assert config.dataset.map is None


@pytest.mark.parametrize(
    ("dataset_type", "change", "match"),
    [
        ("spectrum", ("data", "required_irf", ["aeff", "edisp"]), "requires IRFs"),
        ("spectrum", ("data", "required_irf", ["aeff", "edisp", "rad_max", "bkg"]), "does not use"),
        ("map", ("data", "required_irf", ["aeff", "edisp"]), "requires IRFs"),
        ("spectrum", ("dataset", "unknown", True), "Unknown keys"),
    ],
)
def test_invalid_configuration(raw_config_factory, dataset_type, change, match):
    raw = raw_config_factory(dataset_type)
    section, key, value = change
    raw[section][key] = value

    with pytest.raises(DL4ConfigError, match=match):
        DL4DatasetConfig.parse(raw)


def test_containment_correction_requires_psf(raw_config_factory):
    raw = raw_config_factory()
    raw["dataset"]["spectrum"]["maker"]["containment_correction"] = True

    with pytest.raises(DL4ConfigError, match="requires 'psf'"):
        DL4DatasetConfig.parse(raw)


def test_map_rejects_background_dependent_safe_mask(raw_config_factory):
    raw = raw_config_factory("map")
    raw["dataset"]["safe_mask"] = {"enabled": True, "methods": ["bkg-peak"]}

    with pytest.raises(DL4ConfigError, match="bkg-peak"):
        DL4DatasetConfig.parse(raw)


def test_invalid_energy_bounds(raw_config_factory):
    raw = raw_config_factory()
    raw["dataset"]["energy"]["max"] = raw["dataset"]["energy"]["min"]

    with pytest.raises(DL4ConfigError, match="must be greater"):
        DL4DatasetConfig.parse(raw)


def test_invalid_exclusion_region(raw_config_factory):
    raw = raw_config_factory()
    raw["dataset"]["spectrum"]["background"]["exclusion"] = {"regions": ["not a region"]}

    with pytest.raises(DL4ConfigError, match="Invalid DS9"):
        DL4DatasetConfig.parse(raw)


def test_events_and_gti_are_not_required_irf_entries(raw_config_factory):
    raw = raw_config_factory()
    raw["data"]["required_irf"].append("events")

    with pytest.raises(DL4ConfigError, match="Unknown required_irf"):
        DL4DatasetConfig.parse(raw)


def test_config_does_not_mutate_input(raw_config_factory):
    raw = raw_config_factory()
    original = copy.deepcopy(raw)

    DL4DatasetConfig.parse(raw)

    assert raw == original
