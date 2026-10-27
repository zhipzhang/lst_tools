from pathlib import Path

import pytest

from lst_tools.event_filter import EventFilter
from lst_tools.pipeline import (
    DL1ProducerConfig,
    DL2ProducerConfig,
    DL3ProducerConfig,
    PipelineConfig,
)
from lst_tools.pipeline.configs import ConfigError

RF_DIRECTORY = "/fefs/aswg/data/models/AllSky/nsb_tuning_0.24/dec_347"


class TestDL1Config:
    def test_missing_section_defaults_to_enabled_link(self):
        config = DL1ProducerConfig.parse(None)
        assert config.enabled is True
        assert config.source == "link"
        assert config.link is None

    def test_link_output_directory(self):
        config = DL1ProducerConfig.parse({"link": {"output_directory": "/data/my/dl1"}})
        assert config.link.output_directory == Path("/data/my/dl1")

    def test_build_source_rejected(self):
        with pytest.raises(ConfigError, match="not supported yet"):
            DL1ProducerConfig.parse({"source": "build"})

    def test_unknown_key_rejected(self):
        with pytest.raises(ConfigError, match="Unknown keys"):
            DL1ProducerConfig.parse({"whatever": True})


class TestDL2Config:
    def test_missing_section_means_disabled(self):
        config = DL2ProducerConfig.parse(None)
        assert config.enabled is False

    def test_build_mode(self):
        config = DL2ProducerConfig.parse(
            {
                "source": "build",
                "build": {
                    "output_directory": "/data/my/dl2",
                    "rf_directory": RF_DIRECTORY,
                    "overwrite": True,
                },
            }
        )
        assert config.enabled is True
        assert config.source == "build"
        assert config.build.output_directory == Path("/data/my/dl2")
        assert config.build.rf_directory == RF_DIRECTORY
        assert config.build.overwrite is True

    def test_rf_directory_template_with_nsb_placeholder_is_accepted(self):
        config = DL2ProducerConfig.parse(
            {"source": "build", "build": {"rf_directory": "/models/AllSky/nsb_tuning_$NSB_LEVEL/dec_347"}}
        )
        assert config.build.rf_directory == "/models/AllSky/nsb_tuning_$NSB_LEVEL/dec_347"

    def test_link_output_directory(self):
        config = DL2ProducerConfig.parse({"source": "link", "link": {"output_directory": "/data/my/dl2"}})
        assert config.link.output_directory == Path("/data/my/dl2")

    def test_link_mode(self):
        config = DL2ProducerConfig.parse({"source": "link"})
        assert config.source == "link"
        assert config.build is None

    def test_source_is_required(self):
        with pytest.raises(ConfigError, match='requires source = "build" or "link"'):
            DL2ProducerConfig.parse({"build": {"rf_directory": RF_DIRECTORY}})

    def test_build_mode_requires_build_section(self):
        with pytest.raises(ConfigError, match=r"\[dl2.build\] is required"):
            DL2ProducerConfig.parse({"source": "build"})

    def test_build_mode_requires_rf_directory(self):
        with pytest.raises(ConfigError, match="rf_directory"):
            DL2ProducerConfig.parse({"source": "build", "build": {}})

    def test_build_options_warned_and_skipped_in_link_mode(self):
        with pytest.warns(UserWarning, match=r'\[dl2.build\] options are ignored because source = "link"'):
            config = DL2ProducerConfig.parse({"source": "link", "build": {"rf_directory": RF_DIRECTORY}})
        assert config.build is None

    def test_unknown_link_option_rejected(self):
        with pytest.raises(ConfigError, match=r"Unknown keys in \[dl2.link\]"):
            DL2ProducerConfig.parse({"source": "link", "link": {"rf_directory": RF_DIRECTORY}})

    def test_disabled_skips_mode_validation(self):
        config = DL2ProducerConfig.parse({"enabled": False, "source": "build"})
        assert config.enabled is False
        assert config.build is None


class TestDL3Config:
    def test_missing_section_means_disabled(self):
        config = DL3ProducerConfig.parse(None)
        assert config.enabled is False

    def test_build_mode_requires_build_section(self):
        with pytest.raises(ConfigError, match=r"\[dl3.build\] is required"):
            DL3ProducerConfig.parse({"source": "build"})

    def test_build_mode_requires_irf_section(self):
        with pytest.raises(ConfigError, match=r"\[dl3.build.irf\] is required"):
            DL3ProducerConfig.parse({"source": "build", "build": {"overwrite": True}})

    def test_build_mode_with_irf_section(self):
        config = DL3ProducerConfig.parse(
            {
                "source": "build",
                "build": {
                    "interp_method": "nearest",
                    "use_nearest_irf_node": True,
                    "overwrite": True,
                    "irf": {
                        "intensity_cut": 80,
                        "leakage_cut": 1.0,
                        "gh_efficiency": 0.7,
                        "point_like": True,
                        "mc_dl2_directory": "custom_nsb_$NSB_LEVEL",
                    },
                },
            }
        )
        assert config.source == "build"
        assert config.build.interp_method == "nearest"
        assert config.build.use_nearest_irf_node is True
        assert config.build.overwrite is True
        assert config.build.irf_directory is None
        assert config.build.point_like is True
        assert config.build.mc_dl2_directory == "custom_nsb_$NSB_LEVEL"
        assert isinstance(config.build.event_filter, EventFilter)

    def test_build_irf_defaults_to_full_enclosure(self):
        config = DL3ProducerConfig.parse(
            {
                "source": "build",
                "build": {"irf": {"intensity_cut": 80, "leakage_cut": 1.0, "gh_efficiency": 0.7}},
            }
        )
        assert config.build.point_like is False
        assert config.build.mc_dl2_directory == "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_$NSB_LEVEL"

    def test_irf_section_requires_intensity_and_leakage(self):
        with pytest.raises(ConfigError, match="missing required key"):
            DL3ProducerConfig.parse({"source": "build", "build": {"irf": {"gh_efficiency": 0.7}}})

    def test_irf_section_requires_exactly_one_gh_mode(self):
        irf = {"intensity_cut": 80, "leakage_cut": 1.0, "gh_cut": 0.5, "gh_efficiency": 0.7}
        with pytest.raises(ConfigError, match="exactly one"):
            DL3ProducerConfig.parse({"source": "build", "build": {"irf": irf}})

    def test_build_options_warned_and_skipped_in_link_mode(self):
        with pytest.warns(UserWarning, match=r'\[dl3.build\] options are ignored because source = "link"'):
            config = DL3ProducerConfig.parse({"source": "link", "build": {"overwrite": True}})
        assert config.build is None

    def test_link_options_warned_and_skipped_in_build_mode(self):
        with pytest.warns(UserWarning, match=r'\[dl3.link\] options are ignored because source = "build"'):
            config = DL3ProducerConfig.parse(
                {
                    "source": "build",
                    "link": {"irf_type": "point"},
                    "build": {"irf": {"intensity_cut": 80, "leakage_cut": 1.0, "gh_efficiency": 0.7}},
                }
            )
        assert config.link is None
        assert config.build.event_filter is not None

    def test_link_options_default_to_both_irf_types(self):
        config = DL3ProducerConfig.parse({"source": "link", "link": {}})
        assert config.link.irf_type is None

    @pytest.mark.parametrize("irf_type", ["full-enclosure", "point"])
    def test_link_options_accept_valid_irf_type(self, irf_type):
        config = DL3ProducerConfig.parse({"source": "link", "link": {"irf_type": irf_type}})
        assert config.link.irf_type == irf_type

    def test_link_options_reject_invalid_irf_type(self):
        with pytest.raises(ConfigError, match="irf_type must be"):
            DL3ProducerConfig.parse({"source": "link", "link": {"irf_type": "foo"}})

    def test_disabled_skips_mode_validation(self):
        config = DL3ProducerConfig.parse({"enabled": False, "source": "build"})
        assert config.enabled is False


class TestParsePipelineConfig:
    def test_full_config(self):
        pipeline = PipelineConfig.parse(
            {
                "dl1": {},
                "dl2": {"source": "build", "build": {"rf_directory": RF_DIRECTORY}},
                "dl3": {"source": "link", "link": {"irf_type": "point"}},
            }
        )
        assert pipeline.dl1.enabled is True
        assert pipeline.dl2.source == "build"
        assert pipeline.dl3.link.irf_type == "point"

    def test_empty_config_disables_dl2_and_dl3(self):
        pipeline = PipelineConfig.parse({})
        assert pipeline.dl1.enabled is True
        assert pipeline.dl2.enabled is False
        assert pipeline.dl3.enabled is False

    def test_top_level_irf_section_raises_moved_error(self):
        with pytest.raises(ConfigError, match=r"moved under \[dl3.build.irf\]"):
            PipelineConfig.parse({"irf": {"intensity_cut": 80}})
